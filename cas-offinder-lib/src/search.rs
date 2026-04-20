use opencl3::*;
// use cl3
use crate::bit4ops::{cdiv, roundup};
use crate::run_config::*;
use crate::{bit4_to_string, chrom_chunk::*, reverse_compliment_char_i};
use opencl3::Result;
use std::ptr::null_mut;
use std::sync::mpsc;
use std::sync::Arc;
use std::thread::{self, JoinHandle};

pub const KERNEL_CONTENTS: &str = include_str!("./kernel.cl");
pub const KERNEL_MYERS_CONTENTS: &str = include_str!("./kernel_myers.cl");

// Search chunk size (nucleotides per GPU batch) is chosen at runtime based on
// available device memory — see [`init_search_chunk_nucl`]. The default below
// is used only for CPU-only runs (no OpenCL device query).
const DEFAULT_SEARCH_CHUNK_SIZE: usize = 1 << 22; // must be less than 1<<32
const MAX_SEARCH_CHUNK_SIZE: usize = 1 << 30; // hard cap: 1 Gnt (<1<<32)

static SEARCH_CHUNK_NUCL: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(DEFAULT_SEARCH_CHUNK_SIZE);

#[inline]
fn search_chunk_nucl() -> usize {
    SEARCH_CHUNK_NUCL.load(std::sync::atomic::Ordering::Relaxed)
}
#[inline]
fn search_chunk_bytes() -> usize {
    search_chunk_nucl() / 2
}
#[inline]
fn chunks_per_search() -> usize {
    search_chunk_nucl() / CHUNK_SIZE
}

/// Set the search chunk size based on the largest OpenCL device's memory budget.
/// Picks ~1/4 of CL_DEVICE_MAX_MEM_ALLOC_SIZE per device (leaving room for the
/// output/PEQ/PAM buffers), rounded down to a multiple of CHUNK_SIZE, and
/// clamped to [DEFAULT_SEARCH_CHUNK_SIZE, MAX_SEARCH_CHUNK_SIZE].
pub fn init_search_chunk_nucl_from_devices(devices: &OclRunConfig) {
    let mut best_bytes: u64 = 0;
    for (_, devs) in devices.get().iter() {
        for d in devs {
            if let Ok(alloc_bytes) = d.max_mem_alloc_size() {
                if alloc_bytes > best_bytes {
                    best_bytes = alloc_bytes;
                }
            }
        }
    }
    if best_bytes == 0 {
        // No devices (CPU-only run) — keep default.
        return;
    }
    // Budget per genome buffer: 1/4 of max_alloc, bit4-packed (2 nucl/byte).
    let budget_bytes = (best_bytes / 4) as usize;
    let nucl_budget = budget_bytes.saturating_mul(2);
    // Round down to CHUNK_SIZE multiple.
    let mut size = (nucl_budget / CHUNK_SIZE) * CHUNK_SIZE;
    size = size.clamp(DEFAULT_SEARCH_CHUNK_SIZE, MAX_SEARCH_CHUNK_SIZE);
    SEARCH_CHUNK_NUCL.store(size, std::sync::atomic::Ordering::Relaxed);
}

const CPU_BLOCK_SIZE: usize = 8;
const GPU_BLOCK_SIZE: usize = 4;
const PATTERN_CHUNK_SIZE: usize = 16;
const CL_BLOCKS_PER_EXEC: usize = 4;

struct SearchChunkMeta {
    pub chr_names: Vec<String>,
    // start and end of data within chromosome, by nucleotide
    pub chunk_starts: Vec<u64>,
    pub chunk_ends: Vec<u64>,
    // Byte offset of each chunk inside `SearchChunkInfo.data`. Tight-packed,
    // with entry k holding the start of chunk k and one trailing sentinel at
    // chr_names.len() marking the end of all valid bytes. A chunk whose valid
    // nucleotide count is odd occupies ceil(N/2) bytes; the unused high nibble
    // of its last byte is 0 (bit4 NUL) and gets rejected by the DP.
    pub chunk_byte_offsets: Vec<u32>,
    // True when chunk_buf had `chunks_per_search()` items (full batch); the
    // last chunk is overlap that will be re-sent as the first of the next
    // batch, so matches starting in it must not be emitted here.
    pub has_overlap_tail: bool,
    // True when the first chunk is carried over from the previous batch as
    // backward-context (so a text window near the start of chunk 1 can read
    // into chunk 0 without falling off the search buffer). Matches whose
    // end_pos lies in chunk 0 must not be emitted from this batch — they were
    // already iterated in the previous batch where chunk 0 was the active tail.
    pub has_overlap_head: bool,
}

struct SearchChunkInfo {
    // data length = search_chunk_bytes() (runtime-determined to fit GPU memory)
    pub data: Box<[u8]>,
    pub meta: SearchChunkMeta,
}
struct SearchChunkResult {
    pub matches: Vec<SearchMatch>,
    pub meta: SearchChunkMeta,
    pub data: Box<[u8]>,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct SearchMatch {
    pub chunk_idx: u32,
    pub pattern_idx: u32,
    pub mismatches: u32,
    pub dna_bulge_size: u16,
    pub rna_bulge_size: u16,
}

/// GPU-side output of the Myers+DP+traceback kernel. Each record carries
/// the aligned ops packed as 2 bits per op (LSB = first op emitted during
/// walk-back, i.e. the last op in alignment order). Host walks ops in
/// reverse to rebuild pattern_aligned / text_aligned strings without a DP.
/// Layout (32 B, natural u64-aligned):
///   u32 chunk_idx        -> match_start on genome
///   u32 pattern_idx
///   u64 ops_packed
///   u32 mismatches
///   u16 dna_bulge_size
///   u16 rna_bulge_size
///   u32 ops_count
///   u32 _pad
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
struct GpuEnumMatch {
    pub chunk_idx: u32,
    pub pattern_idx: u32,
    pub ops_packed: u64,
    pub mismatches: u32,
    pub dna_bulge_size: u16,
    pub rna_bulge_size: u16,
    pub ops_count: u32,
    pub _pad: u32,
}

const GPU_OP_MATCH: u8 = 0;
const GPU_OP_SUB: u8 = 1;
const GPU_OP_DNA: u8 = 2;
const GPU_OP_RNA: u8 = 3;

/// Decode a [`GpuEnumMatch`] into a (SearchMatch, MyersAlignment) pair by
/// walking the packed ops (in reverse of traceback order = forward alignment
/// order) and pulling the corresponding characters from pattern + genome.
///
/// Returns `None` if the alignment touches a padding region between concatenated
/// chromosome chunks (NUL / 0x00 in the decoded ASCII buffer). The GPU kernel
/// is chromosome-unaware — it scans a contiguous search buffer that includes
/// trailing zero-padding after each partial chromosome chunk — so a valid
/// Myers candidate can technically land where one strand of the alignment
/// crosses into NUL territory. Such matches are not real hits and get dropped
/// here.
fn decode_gpu_enum_match(
    raw: &GpuEnumMatch,
    ascii_chunk: &[u8],
    pattern_ascii: &[u8],
) -> Option<(SearchMatch, MyersAlignment)> {
    let ops_count = raw.ops_count as usize;
    let mut pattern_aligned: Vec<u8> = Vec::with_capacity(ops_count);
    let mut text_aligned: Vec<u8> = Vec::with_capacity(ops_count);
    let mut pi: usize = 0;
    let mut gi: usize = raw.chunk_idx as usize;
    for k_rev in (0..ops_count).rev() {
        let op = ((raw.ops_packed >> (k_rev * 2)) & 0x3) as u8;
        match op {
            GPU_OP_MATCH | GPU_OP_SUB => {
                let ch = ascii_chunk[gi];
                if ch == 0 {
                    return None;
                }
                pattern_aligned.push(pattern_ascii[pi]);
                text_aligned.push(ch);
                pi += 1;
                gi += 1;
            }
            GPU_OP_DNA => {
                let ch = ascii_chunk[gi];
                if ch == 0 {
                    return None;
                }
                pattern_aligned.push(b'-');
                text_aligned.push(ch);
                gi += 1;
            }
            GPU_OP_RNA => {
                pattern_aligned.push(pattern_ascii[pi]);
                text_aligned.push(b'-');
                pi += 1;
            }
            _ => unreachable!("invalid op code {}", op),
        }
    }
    Some((
        SearchMatch {
            chunk_idx: raw.chunk_idx,
            pattern_idx: raw.pattern_idx,
            mismatches: raw.mismatches,
            dna_bulge_size: raw.dna_bulge_size,
            rna_bulge_size: raw.rna_bulge_size,
        },
        MyersAlignment {
            pattern_aligned,
            text_aligned,
        },
    ))
}

const MAX_QUEUED: usize = 1;
unsafe fn create_ocl_buf<T>(context: &context::Context, size: usize) -> Result<memory::Buffer<T>> {
    memory::Buffer::create(context, memory::CL_MEM_READ_WRITE, size, null_mut())
}
unsafe fn create_ocl_bufs<T>(
    context: &context::Context,
    size: usize,
) -> Result<[memory::Buffer<T>; MAX_QUEUED]> {
    Ok([
        create_ocl_buf::<T>(context, size)?,
        // create_ocl_buf::<T>(&context,size)?
    ])
}
fn is_gpu(dev: &device::Device) -> Result<bool> {
    Ok(dev.dev_type()? == device::CL_DEVICE_TYPE_GPU)
}
fn prefered_block_size(dev: &device::Device) -> Result<usize> {
    Ok(if is_gpu(dev)? {
        GPU_BLOCK_SIZE
    } else {
        CPU_BLOCK_SIZE
    })
}
// fn get_prog_args(pattern_len:)
fn search_device_ocl(
    max_mismatches: u32,
    pattern_len: usize,
    patterns: Arc<Vec<u8>>,
    context: Arc<context::Context>,
    program: Arc<program::Program>,
    dev: Arc<device::Device>,
    recv: crossbeam_channel::Receiver<SearchChunkInfo>,
    dest: mpsc::SyncSender<SearchChunkResult>,
) -> Result<()> {
    unsafe {
        const OUT_BUF_SIZE: usize = 1 << 22;
        const CL_BLOCK: u32 = 1;
        const CL_NO_BLOCK: u32 = 0;
        let queue = command_queue::CommandQueue::create(&context, dev.id(), 0)?;
        let kernel = kernel::Kernel::create(&program, "find_matches")?;
        let mut genome_bufs = create_ocl_bufs::<u8>(&context, search_chunk_bytes())?;
        let mut out_counts = create_ocl_bufs::<u32>(&context, 1)?;
        let mut out_bufs = create_ocl_bufs::<SearchMatch>(&context, OUT_BUF_SIZE)?;
        let mut pattern_buf = create_ocl_buf::<u8>(&context, patterns.len())?;
        queue.enqueue_write_buffer(&mut pattern_buf, CL_BLOCK, 0, &patterns, &[])?;
        let pattern_blocked_size = roundup(cdiv(pattern_len, 2), PATTERN_CHUNK_SIZE);
        assert!(patterns.len() % pattern_blocked_size == 0);
        let n_patterns = patterns.len() / pattern_blocked_size;
        for item in recv.iter() {
            let total_chunks = item.meta.chr_names.len();
            let n_chunks = if item.meta.has_overlap_tail {
                total_chunks - 1
            } else {
                total_chunks
            };
            let n_genome_bytes = item.meta.chunk_byte_offsets[n_chunks] as usize;
            let n_total_bytes =
                *item.meta.chunk_byte_offsets.last().unwrap() as usize;
            let n_genome_blocks = n_genome_bytes / prefered_block_size(&dev)?;
            let n_genome_execs = n_genome_blocks / CL_BLOCKS_PER_EXEC;
            let cur_genome_buf = &mut genome_bufs[0];
            let cur_size_buf = &mut out_counts[0];
            let cur_out_buf = &mut out_bufs[0];
            let write_event = queue.enqueue_write_buffer(
                cur_genome_buf,
                CL_NO_BLOCK,
                0,
                &item.data[..n_total_bytes],
                &[],
            )?;
            let clear_count_event =
                queue.enqueue_write_buffer(cur_size_buf, CL_NO_BLOCK, 0, &[0], &[])?;

            let kernel_event = kernel::ExecuteKernel::new(&kernel)
                .set_arg(cur_genome_buf)
                .set_arg(&pattern_buf)
                .set_arg(&max_mismatches)
                .set_arg(cur_out_buf)
                .set_arg(cur_size_buf)
                .set_global_work_sizes(&[n_genome_execs, n_patterns])
                .set_wait_event(&write_event)
                .set_wait_event(&clear_count_event)
                .enqueue_nd_range(&queue)?;
            let mut readsize_buf = [0];
            queue.enqueue_read_buffer(
                cur_size_buf,
                CL_BLOCK,
                0,
                &mut readsize_buf,
                &[kernel_event.get()],
            )?;
            let readsize = readsize_buf[0];
            if readsize != 0 {
                let mut outvec: Vec<SearchMatch> = vec![
                    SearchMatch {
                        chunk_idx: 0,
                        pattern_idx: 0,
                        mismatches: 0,
                        dna_bulge_size: 0,
                        rna_bulge_size: 0,
                    };
                    readsize as usize
                ];
                queue.enqueue_read_buffer(cur_out_buf, CL_BLOCK, 0, &mut outvec[..], &[])?;
                dest.send(SearchChunkResult {
                    matches: outvec,
                    meta: item.meta,
                    data: item.data,
                })
                .unwrap();
            }
        }
    }
    Ok(())
}
fn prefered_block_type(dev: &device::Device) -> Result<&str> {
    Ok(if is_gpu(dev)? { "uint32_t" } else { "uint64_t" })
}
fn get_compile_defs(pattern_len: usize, block_ty: &str) -> String {
    format!(
        " -DPATTERN_LEN={}
     -DBLOCKS_PER_EXEC={}
     -DPATTERN_CHUNK_SIZE={}
     -Dblock_ty={}",
        pattern_len, CL_BLOCKS_PER_EXEC, PATTERN_CHUNK_SIZE, block_ty
    )
}
fn search_chunk_ocl(
    devices: OclRunConfig,
    max_mismatches: u32,
    pattern_len: usize,
    patterns: &[Vec<u8>],
    recv: crossbeam_channel::Receiver<SearchChunkInfo>,
    dest: mpsc::SyncSender<SearchChunkResult>,
) -> Result<()> {
    /* divies off work to opencl devices */
    let pattern_arc = Arc::new(pack_patterns(patterns));
    // let devices = get_all_devices()?;
    // assert!(devices.len()>0, "Needs at least one opencl device to run tests!");
    let mut threads: Vec<JoinHandle<Result<()>>> = Vec::new();
    for (_, devs) in devices.get().iter() {
        let plat_devs: Vec<*mut std::ffi::c_void> = devs.iter().map(|d| d.id()).collect();
        if !plat_devs.is_empty() {
            let context = Arc::new(context::Context::from_devices(
                &plat_devs,
                &[0],
                None,
                null_mut(),
            )?);
            let p_devices: Vec<Arc<device::Device>> = plat_devs
                .iter()
                .map(|d| Arc::new(device::Device::new(*d)))
                .collect();
            let prog_options = get_compile_defs(pattern_len, prefered_block_type(&p_devices[0])?);
            let program = Arc::new(
                program::Program::create_and_build_from_source(
                    &context,
                    KERNEL_CONTENTS,
                    &prog_options,
                )
                .map_err(|err| {
                    eprintln!("{}", err);
                })
                .unwrap(),
            );

            for p_dev in p_devices {
                let t_dest = dest.clone();
                let t_recv = recv.clone();
                let t_context = context.clone();
                let t_prog = program.clone();
                let t_pattern = pattern_arc.clone();
                threads.push(thread::spawn(move || {
                    search_device_ocl(
                        max_mismatches,
                        pattern_len,
                        t_pattern,
                        t_context,
                        t_prog,
                        p_dev,
                        t_recv,
                        t_dest,
                    )
                }));
            }
        }
    }
    for t in threads {
        t.join().unwrap()?;
    }
    Ok(())
}
fn checked_div(x: usize, y: usize) -> usize {
    assert!(x % y == 0);
    x / y
}
fn pack(d: &[u8]) -> u64 {
    assert!(d.len() == 8);
    unsafe { std::mem::transmute::<[u8; 8], u64>([d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7]]) }
}
fn block_data_cpu(data: &[u8]) -> Vec<u64> {
    data.chunks(8).map(pack).collect()
}
fn search_chunk_cpu(
    max_mismatches: u32,
    pattern_len: usize,
    packed_patterns: &[u8],
    data: &[u8],
) -> Vec<SearchMatch> {
    let mut matches: Vec<SearchMatch> = Vec::new();
    assert!(
        data.as_ptr().align_offset(8) == 0,
        "data must be 8 byte aligned"
    );
    let n_patterns = checked_div(
        packed_patterns.len(),
        roundup(pattern_len, 2 * PATTERN_CHUNK_SIZE) / 2,
    );
    let pattern_ptr = block_data_cpu(packed_patterns); //packed_patterns.as_ptr() as *const u64;
    let genome_ptr = block_data_cpu(data); //data.as_ptr() as *const u64;
    let genome_blocks = checked_div(data.len(), CPU_BLOCK_SIZE);
    let pattern_blocks = cdiv(pattern_len, CPU_BLOCK_SIZE * 2);
    let packed_pattern_size = roundup(pattern_blocks, PATTERN_CHUNK_SIZE / CPU_BLOCK_SIZE);
    // assert!(pattern_blocks == 2);
    // assert!(n_patterns == 2);
    const NUCL_PER_BLOCK: usize = 2 * std::mem::size_of::<u64>();
    const BLOCKS_PER_EXEC: usize = 4;
    let mut shifted_data = vec![0_u64; BLOCKS_PER_EXEC + pattern_blocks + 1];
    for gen_block_idx in 0..checked_div(genome_blocks, BLOCKS_PER_EXEC) {
        let gen_idx = gen_block_idx * BLOCKS_PER_EXEC;
        shifted_data.fill(0);
        let n_copy = std::cmp::min(
            BLOCKS_PER_EXEC + pattern_blocks + 1,
            genome_blocks - gen_idx,
        );
        shifted_data[..n_copy].copy_from_slice(&genome_ptr[gen_idx..][..n_copy]);
        for l in 0..NUCL_PER_BLOCK {
            for j in 0..n_patterns {
                let mut num_matches = [0_u32; BLOCKS_PER_EXEC];
                for k in 0..pattern_blocks {
                    for o in 0..BLOCKS_PER_EXEC {
                        num_matches[o] += (shifted_data[k + o]
                            & pattern_ptr[j * packed_pattern_size + k])
                            .count_ones()
                    }
                }
                for o in 0..BLOCKS_PER_EXEC {
                    let mismatches = pattern_len as u32 - num_matches[o];
                    if mismatches <= max_mismatches {
                        matches.push(SearchMatch {
                            chunk_idx: ((gen_idx + o) * NUCL_PER_BLOCK + l) as u32,
                            pattern_idx: j as u32,
                            mismatches: mismatches,
                            dna_bulge_size: 0,
                            rna_bulge_size: 0,
                        });
                    }
                }
            }
            for k in 0..(pattern_blocks + BLOCKS_PER_EXEC - 1) {
                shifted_data[k] >>= 4;
                shifted_data[k] |= shifted_data[k + 1] << (4 * (NUCL_PER_BLOCK - 1));
            }
            shifted_data[pattern_blocks + BLOCKS_PER_EXEC - 1] >>= 4;
        }
    }
    matches
}

fn pack_patterns(patterns: &[Vec<u8>]) -> Vec<u8> {
    patterns
        .iter()
        .flat_map(|pattern| {
            let pattern_padding =
                cdiv(pattern.len(), PATTERN_CHUNK_SIZE) * PATTERN_CHUNK_SIZE - pattern.len();
            pattern
                .iter()
                .copied()
                .chain((0..pattern_padding).map(|_| 0_u8))
        })
        .collect()
}
fn search_device_cpu_thread(
    max_mismatches: u32,
    pattern_len: usize,
    packed_patterns: Arc<Vec<u8>>,
    recv: crossbeam_channel::Receiver<SearchChunkInfo>,
    dest: mpsc::SyncSender<SearchChunkResult>,
) {
    for schunk in recv.iter() {
        dest.send(SearchChunkResult {
            matches: search_chunk_cpu(max_mismatches, pattern_len, &packed_patterns, &schunk.data),
            meta: schunk.meta,
            data: schunk.data,
        })
        .unwrap();
    }
}
fn search_compute_cpu(
    max_mismatches: u32,
    pattern_len: usize,
    patterns: &[Vec<u8>],
    recv: crossbeam_channel::Receiver<SearchChunkInfo>,
    dest: mpsc::SyncSender<SearchChunkResult>,
) {
    /* divies off work to threads devices */
    let pattern_arc = Arc::new(pack_patterns(patterns));
    let n_threads = thread::available_parallelism().unwrap().into();
    let mut threads: Vec<thread::JoinHandle<()>> = Vec::new();
    for _ in 0..n_threads {
        let tpattners = pattern_arc.clone();
        let trecv = recv.clone();
        let tdest = dest.clone();
        threads.push(thread::spawn(move || {
            search_device_cpu_thread(max_mismatches, pattern_len, tpattners, trecv, tdest)
        }));
    }
    for t in threads {
        t.join().unwrap();
    }
}

// ============================================================================
// Myers bit-parallel CPU path (supports DNA + RNA bulges)
// ============================================================================

/// Per-match extra alignment info from Myers path, kept parallel to SearchMatch.
#[derive(Clone, Debug)]
pub struct MyersAlignment {
    pub pattern_aligned: Vec<u8>, // pattern bases with '-' for DNA bulge positions
    pub text_aligned: Vec<u8>,    // text bases with '-' for RNA bulge positions
}

struct SearchChunkResultMyers {
    pub matches: Vec<SearchMatch>,
    pub alignments: Vec<MyersAlignment>,
    pub meta: SearchChunkMeta,
    pub data: Box<[u8]>,
}

fn nucl_idx_ascii(c: u8) -> u8 {
    match c.to_ascii_uppercase() {
        b'A' => 0,
        b'C' => 1,
        b'G' => 2,
        b'T' => 3,
        _ => 4,
    }
}

/// Given a Myers candidate (alignment ending at `end_pos` in `ascii_chunk`),
/// enumerate every valid traceback. Returns one (SearchMatch, MyersAlignment)
/// per distinct bulge placement / mismatch set that satisfies all per-type caps
/// and the PAM constraint — matching original cas-offinder's behavior of
/// emitting each alternative placement as its own hit.
#[allow(clippy::too_many_arguments)]
fn classify_myers_candidate(
    end_pos: usize,
    pattern_idx: usize,
    ascii_chunk: &[u8],
    patterns_ascii: &[Vec<u8>],
    pattern_is_n: &[Vec<bool>],
    effective_filters_bit4: &[Vec<u8>],
    max_mismatches: u32,
    max_dna_bulges: u32,
    max_rna_bulges: u32,
    text_window_len: usize,
) -> Vec<(SearchMatch, MyersAlignment)> {
    use crate::traceback::{traceback_all, EditOp};

    let max_edits = max_mismatches + max_dna_bulges + max_rna_bulges;
    let pattern = &patterns_ascii[pattern_idx];
    let is_n_pat = &pattern_is_n[pattern_idx];
    let eff_filter = &effective_filters_bit4[pattern_idx];

    let text_start_in_chunk = (end_pos + 1).saturating_sub(text_window_len);
    let text = &ascii_chunk[text_start_in_chunk..=end_pos];

    let alignments = traceback_all(
        pattern,
        text,
        max_edits,
        max_dna_bulges,
        max_rna_bulges,
        max_mismatches,
        is_n_pat,
    );
    let mut out: Vec<(SearchMatch, MyersAlignment)> = Vec::with_capacity(alignments.len());

    'alignments: for align in alignments {
        // PAM check: for each pattern position that maps to a genome base,
        // verify the effective filter matches. (traceback_all already pruned
        // edits touching PAM positions, so mm/db/rb counts are valid.)
        let genome_span: usize = align.text_aligned.iter().filter(|&&c| c != b'-').count();
        let align_text_offset = text.len() - genome_span;
        let mut p_in_pat = 0usize;
        let mut g_off = align_text_offset;
        for op in &align.ops {
            match op {
                EditOp::Match | EditOp::Substitution => {
                    let genome_c = text[g_off];
                    let g_bit4 = crate::bit4ops::char_to_bit4(genome_c);
                    let f_bit4 = eff_filter[p_in_pat];
                    if (g_bit4 & f_bit4) == 0 {
                        continue 'alignments;
                    }
                    p_in_pat += 1;
                    g_off += 1;
                }
                EditOp::DnaBulge => {
                    g_off += 1;
                }
                EditOp::RnaBulge => {
                    p_in_pat += 1;
                }
            }
        }

        let match_start = end_pos + 1 - genome_span;
        out.push((
            SearchMatch {
                chunk_idx: match_start as u32,
                pattern_idx: pattern_idx as u32,
                mismatches: align.mismatches,
                dna_bulge_size: align.dna_bulges as u16,
                rna_bulge_size: align.rna_bulges as u16,
            },
            MyersAlignment {
                pattern_aligned: align.pattern_aligned,
                text_aligned: align.text_aligned,
            },
        ));
    }
    out
}

/// Quick PAM pre-check: given an alignment ending at `end_pos` (inclusive),
/// verify that the PAM positions in the genome match the expected filter.
/// Assumes no bulge in PAM region (which we enforce anyway).
/// `pam_offset`: distance from alignment start to PAM start in pattern
/// `pam_filter`: bit4 filter values for the PAM positions only (length = pam_len)
///
/// DNA bulges shift the alignment start leftward. For PAM-first patterns
/// (pam_offset == 0), PAM position depends on the actual bulge count in this
/// candidate, which we don't know yet. So we try each possible bulge count in
/// [0, max_dna_bulges] and accept if any shift yields a valid PAM.
/// For PAM-last patterns, PAM is anchored to end_pos and unaffected by bulges.
fn check_pam_quick(
    ascii_chunk: &[u8],
    end_pos: usize,
    pattern_len: usize,
    pam_offset: usize,
    pam_filter: &[u8],
    max_dna_bulges: u32,
) -> bool {
    let shift_range: usize = if pam_offset == 0 {
        max_dna_bulges as usize + 1
    } else {
        1
    };
    for b in 0..shift_range {
        let align_start = match (end_pos + 1).checked_sub(pattern_len + b) {
            Some(x) => x,
            None => continue,
        };
        let mut ok = true;
        for (k, &f) in pam_filter.iter().enumerate() {
            let gpos = align_start + pam_offset + k;
            if gpos >= ascii_chunk.len() {
                ok = false;
                break;
            }
            let g = crate::bit4ops::char_to_bit4(ascii_chunk[gpos]);
            if (g & f) == 0 {
                ok = false;
                break;
            }
        }
        if ok {
            return true;
        }
    }
    false
}

/// Precompute per-pattern PAM pre-check data:
/// Returns (pam_offset, pam_filter_bit4) for each pattern.
/// pam_offset = position in pattern where PAM (N-mask) starts.
/// pam_filter_bit4 = the effective filter bytes at PAM positions only.
fn precompute_pam_precheck(
    pattern_is_n: &[Vec<bool>],
    effective_filters_bit4: &[Vec<u8>],
) -> Vec<(usize, Vec<u8>)> {
    pattern_is_n
        .iter()
        .zip(effective_filters_bit4.iter())
        .map(|(is_n, eff_filter)| {
            let pam_start = is_n.iter().position(|&x| x).unwrap_or(is_n.len());
            let pam_filter: Vec<u8> = (pam_start..is_n.len())
                .filter(|&i| is_n[i])
                .map(|i| eff_filter[i])
                .collect();
            (pam_start, pam_filter)
        })
        .collect()
}

fn search_chunk_myers(
    max_mismatches: u32,
    max_dna_bulges: u32,
    max_rna_bulges: u32,
    patterns_ascii: &[Vec<u8>],
    effective_filters_bit4: &[Vec<u8>],
    pattern_is_n: &[Vec<bool>],
    pam_precheck: &[(usize, Vec<u8>)],
    peqs: &[crate::myers::PeqTable],
    chunk_data_bit4: &[u8],
    active_start_nucl: usize,
    active_end_nucl: usize,
) -> (Vec<SearchMatch>, Vec<MyersAlignment>) {
    let max_edits = max_mismatches + max_dna_bulges + max_rna_bulges;
    let total_nucl = active_end_nucl.min(chunk_data_bit4.len() * 2);
    let pattern_len = patterns_ascii[0].len();
    let text_window_len = pattern_len + max_dna_bulges as usize;

    // Decode only the active (non-padding) region to ASCII
    let mut ascii_chunk = vec![0u8; total_nucl];
    crate::bit4_to_string(&mut ascii_chunk, chunk_data_bit4, 0, total_nucl);

    let mut matches: Vec<SearchMatch> = Vec::new();
    let mut alignments: Vec<MyersAlignment> = Vec::new();

    let mask = if pattern_len < 64 {
        (1u64 << pattern_len) - 1
    } else {
        !0u64
    };
    let last_bit = 1u64 << (pattern_len - 1);

    // Per-position windowed Myers on BIT4, byte-for-byte the same operation
    // as the GPU kernel. CPU and GPU now consume the same encoding (no lossy
    // ASCII round-trip) and produce identical candidate sets.
    let get_bit4 = |i: usize| -> u8 {
        let byte = chunk_data_bit4[i / 2];
        if i % 2 == 0 {
            byte & 0x0F
        } else {
            (byte >> 4) & 0x0F
        }
    };
    for (p_idx, peq) in peqs.iter().enumerate() {
        let (pam_offset, ref pam_filter) = pam_precheck[p_idx];

        for t_pos in active_start_nucl..total_nucl {
            if t_pos + 1 < pattern_len {
                continue;
            }

            // PAM pre-check first (cheap fail-fast, same order as kernel)
            if !check_pam_quick(&ascii_chunk, t_pos, pattern_len, pam_offset, pam_filter, max_dna_bulges) {
                continue;
            }

            // Fresh Myers sweep over the candidate's text window, mirroring
            // the GPU kernel exactly. Bit4 NUL (0) contributes no peq bits.
            let text_start = (t_pos + 1).saturating_sub(text_window_len);
            let mut vp: u64 = mask;
            let mut vn: u64 = 0;
            let mut score: i32 = pattern_len as i32;
            for t in text_start..=t_pos {
                let b4 = get_bit4(t);
                let mut eq: u64 = 0;
                if b4 & 0x4 != 0 { eq |= peq.peq[0]; } // A
                if b4 & 0x2 != 0 { eq |= peq.peq[1]; } // C
                if b4 & 0x8 != 0 { eq |= peq.peq[2]; } // G
                if b4 & 0x1 != 0 { eq |= peq.peq[3]; } // T
                let x = eq | vn;
                let d0 = (((x & vp).wrapping_add(vp)) ^ vp) | x;
                let hn = vp & d0;
                let hp = vn | !(vp | d0);
                let x_shift = hp << 1;
                vn = x_shift & d0;
                vp = (hn << 1) | !(x_shift | d0);
                vp &= mask;
                vn &= mask;
                if hp & last_bit != 0 {
                    score += 1;
                }
                if hn & last_bit != 0 {
                    score -= 1;
                }
            }
            if score < 0 || (score as u32) > max_edits {
                continue;
            }

            for (sm, al) in classify_myers_candidate(
                t_pos,
                p_idx,
                &ascii_chunk,
                patterns_ascii,
                pattern_is_n,
                effective_filters_bit4,
                max_mismatches,
                max_dna_bulges,
                max_rna_bulges,
                text_window_len,
            ) {
                matches.push(sm);
                alignments.push(al);
            }
        }
    }
    (matches, alignments)
}

fn search_device_cpu_thread_myers(
    max_mismatches: u32,
    max_dna_bulges: u32,
    max_rna_bulges: u32,
    patterns_ascii: Arc<Vec<Vec<u8>>>,
    effective_filters_bit4: Arc<Vec<Vec<u8>>>,
    pattern_is_n: Arc<Vec<Vec<bool>>>,
    pam_precheck: Arc<Vec<(usize, Vec<u8>)>>,
    peqs: Arc<Vec<crate::myers::PeqTable>>,
    recv: crossbeam_channel::Receiver<SearchChunkInfo>,
    dest: mpsc::SyncSender<SearchChunkResultMyers>,
) {
    for schunk in recv.iter() {
        let total_chunks = schunk.meta.chr_names.len();
        let n_chunks = if schunk.meta.has_overlap_tail {
            total_chunks - 1
        } else {
            total_chunks
        };
        // Active iteration range: skip the head-overlap chunk (exists only to
        // provide backward text-window context for chunk 1); iterate through
        // the tail-overlap exclusion that has_overlap_tail already applies.
        let active_start_nucl = if schunk.meta.has_overlap_head {
            schunk.meta.chunk_byte_offsets[1] as usize * 2
        } else {
            0
        };
        let active_end_nucl = schunk.meta.chunk_byte_offsets[n_chunks] as usize * 2;
        let (matches, alignments) = search_chunk_myers(
            max_mismatches,
            max_dna_bulges,
            max_rna_bulges,
            &patterns_ascii,
            &effective_filters_bit4,
            &pattern_is_n,
            &pam_precheck,
            &peqs,
            &schunk.data,
            active_start_nucl,
            active_end_nucl,
        );
        dest.send(SearchChunkResultMyers {
            matches,
            alignments,
            meta: schunk.meta,
            data: schunk.data,
        })
        .unwrap();
    }
}

fn search_compute_cpu_myers(
    max_mismatches: u32,
    max_dna_bulges: u32,
    max_rna_bulges: u32,
    search_filter_ascii: &[u8],
    patterns_ascii: &[Vec<u8>],
    n_original_patterns: usize,
    recv: crossbeam_channel::Receiver<SearchChunkInfo>,
    dest: mpsc::SyncSender<SearchChunkResultMyers>,
) {
    use crate::bit4ops::{char_to_bit4, reverse_compliment_char};
    use crate::myers::build_peq;

    // Precompute shared per-pattern tables
    let peqs: Vec<crate::myers::PeqTable> =
        patterns_ascii.iter().map(|p| build_peq(p)).collect();
    let pattern_is_n: Vec<Vec<bool>> = patterns_ascii
        .iter()
        .map(|p| p.iter().map(|&c| c == b'N' || c == b'n').collect())
        .collect();
    let filter_revcomp = reverse_compliment_char(search_filter_ascii);
    let effective_filters_bit4: Vec<Vec<u8>> = patterns_ascii
        .iter()
        .enumerate()
        .map(|(p_idx, _)| {
            let f_ascii: &[u8] = if p_idx < n_original_patterns {
                search_filter_ascii
            } else {
                &filter_revcomp
            };
            f_ascii.iter().map(|&c| char_to_bit4(c)).collect()
        })
        .collect();

    let pam_precheck = precompute_pam_precheck(&pattern_is_n, &effective_filters_bit4);

    let peqs_arc = Arc::new(peqs);
    let is_n_arc = Arc::new(pattern_is_n);
    let filters_arc = Arc::new(effective_filters_bit4);
    let pam_arc = Arc::new(pam_precheck);
    let patterns_arc = Arc::new(patterns_ascii.to_vec());

    let n_threads: usize = thread::available_parallelism().unwrap().into();
    let mut threads: Vec<thread::JoinHandle<()>> = Vec::new();
    for _ in 0..n_threads {
        let t_pat = patterns_arc.clone();
        let t_filt = filters_arc.clone();
        let t_isn = is_n_arc.clone();
        let t_pam = pam_arc.clone();
        let t_peqs = peqs_arc.clone();
        let t_recv = recv.clone();
        let t_dest = dest.clone();
        threads.push(thread::spawn(move || {
            search_device_cpu_thread_myers(
                max_mismatches,
                max_dna_bulges,
                max_rna_bulges,
                t_pat,
                t_filt,
                t_isn,
                t_pam,
                t_peqs,
                t_recv,
                t_dest,
            )
        }));
    }
    for t in threads {
        t.join().unwrap();
    }
}

// ============================================================================
// Myers GPU path
// ============================================================================

fn build_peq_array(peqs: &[crate::myers::PeqTable]) -> Vec<u64> {
    let mut out = Vec::with_capacity(peqs.len() * 4);
    for peq in peqs {
        out.push(peq.peq[0]);
        out.push(peq.peq[1]);
        out.push(peq.peq[2]);
        out.push(peq.peq[3]);
    }
    out
}

#[allow(clippy::too_many_arguments)]
unsafe fn search_device_ocl_myers(
    _max_mismatches: u32,
    _max_dna_bulges: u32,
    _max_rna_bulges: u32,
    peq_array: Arc<Vec<u64>>,
    pattern_bit4_flat: Arc<Vec<u8>>,
    pam_offsets_gpu: Arc<Vec<u8>>,
    pam_filters_flat_gpu: Arc<Vec<u8>>,
    n_patterns: u32,
    _n_fwd_patterns: u32,
    patterns_ascii: Arc<Vec<Vec<u8>>>,
    context: Arc<context::Context>,
    program: Arc<program::Program>,
    _dev: Arc<device::Device>,
    recv: crossbeam_channel::Receiver<SearchChunkInfo>,
    dest: mpsc::SyncSender<SearchChunkResultMyers>,
    out_buf_size: usize,
) -> Result<()> {
    const CL_BLOCK: u32 = 1;
    const CL_NO_BLOCK: u32 = 0;

    let queue = command_queue::CommandQueue::create(&context, _dev.id(), 0)?;
    let kernel = kernel::Kernel::create(&program, "find_matches_myers")?;
    let mut genome_buf = create_ocl_buf::<u8>(&context, search_chunk_bytes())?;
    let mut out_count = create_ocl_buf::<u32>(&context, 1)?;
    let mut out_buf = create_ocl_buf::<GpuEnumMatch>(&context, out_buf_size)?;
    let mut peq_buf = create_ocl_buf::<u64>(&context, peq_array.len())?;
    let mut pattern_buf = create_ocl_buf::<u8>(&context, pattern_bit4_flat.len())?;
    let mut pam_off_buf = create_ocl_buf::<u8>(&context, pam_offsets_gpu.len())?;
    let mut pam_filt_buf = create_ocl_buf::<u8>(&context, pam_filters_flat_gpu.len())?;
    queue.enqueue_write_buffer(&mut peq_buf, CL_BLOCK, 0, &peq_array, &[])?;
    queue.enqueue_write_buffer(&mut pattern_buf, CL_BLOCK, 0, &pattern_bit4_flat, &[])?;
    queue.enqueue_write_buffer(&mut pam_off_buf, CL_BLOCK, 0, &pam_offsets_gpu, &[])?;
    queue.enqueue_write_buffer(&mut pam_filt_buf, CL_BLOCK, 0, &pam_filters_flat_gpu, &[])?;

    for item in recv.iter() {
        // With tight packing, chunks are variable-size. `n_chunks` is the
        // number of "active" chunks (emit matches from these). If this batch
        // carries an overlap-tail chunk, we stop iterating before it.
        let total_chunks = item.meta.chr_names.len();
        let n_chunks = if item.meta.has_overlap_tail {
            total_chunks - 1
        } else {
            total_chunks
        };
        let active_start_nucl: u32 = if item.meta.has_overlap_head {
            item.meta.chunk_byte_offsets[1] as u32 * 2
        } else {
            0
        };
        let n_active_nucl: u32 = item.meta.chunk_byte_offsets[n_chunks] as u32 * 2;
        let n_write_bytes = *item.meta.chunk_byte_offsets.last().unwrap() as usize;

        let write_event = queue.enqueue_write_buffer(
            &mut genome_buf,
            CL_NO_BLOCK,
            0,
            &item.data[..n_write_bytes],
            &[],
        )?;
        let clear_count_event =
            queue.enqueue_write_buffer(&mut out_count, CL_NO_BLOCK, 0, &[0u32], &[])?;

        let kernel_event = kernel::ExecuteKernel::new(&kernel)
            .set_arg(&genome_buf)
            .set_arg(&peq_buf)
            .set_arg(&pattern_buf)
            .set_arg(&pam_off_buf)
            .set_arg(&pam_filt_buf)
            .set_arg(&n_patterns)
            .set_arg(&_n_fwd_patterns)
            .set_arg(&active_start_nucl)
            .set_arg(&n_active_nucl)
            .set_arg(&out_buf)
            .set_arg(&out_count)
            .set_global_work_sizes(&[n_active_nucl as usize, n_patterns as usize])
            .set_wait_event(&write_event)
            .set_wait_event(&clear_count_event)
            .enqueue_nd_range(&queue)?;

        let mut readsize_buf = [0u32; 1];
        queue.enqueue_read_buffer(
            &out_count,
            CL_BLOCK,
            0,
            &mut readsize_buf,
            &[kernel_event.get()],
        )?;
        let total_found = readsize_buf[0] as usize;
        if total_found > out_buf_size {
            panic!(
                "GPU match buffer overflow: {} candidates exceeds out_buf_size={}. \
                 Reduce SEARCH_CHUNK_NUCL or increase the output buffer budget.",
                total_found, out_buf_size
            );
        }
        let readsize = total_found;

        if readsize == 0 {
            dest.send(SearchChunkResultMyers {
                matches: Vec::new(),
                alignments: Vec::new(),
                meta: item.meta,
                data: item.data,
            })
            .unwrap();
            continue;
        }

        let mut raw_matches: Vec<GpuEnumMatch> =
            vec![GpuEnumMatch::default(); readsize];
        queue.enqueue_read_buffer(&out_buf, CL_BLOCK, 0, &mut raw_matches[..], &[])?;

        // Host post-process: decode packed ops into aligned strings using
        // pattern_ascii + genome ASCII. No DP — just bit unpacking and
        // char gather. rayon parallelizes across cores.
        let active_nucl = n_active_nucl as usize;
        let mut ascii_chunk = vec![0u8; active_nucl];
        crate::bit4_to_string(&mut ascii_chunk, &item.data[..], 0, active_nucl);

        use rayon::prelude::*;
        let decoded: Vec<(SearchMatch, MyersAlignment)> = raw_matches
            .par_iter()
            .filter_map(|raw| {
                let pattern_ascii = &patterns_ascii[raw.pattern_idx as usize];
                decode_gpu_enum_match(raw, &ascii_chunk, pattern_ascii)
            })
            .collect();
        let (final_matches, alignments): (Vec<_>, Vec<_>) = decoded.into_iter().unzip();

        dest.send(SearchChunkResultMyers {
            matches: final_matches,
            alignments,
            meta: item.meta,
            data: item.data,
        })
        .unwrap();
    }
    Ok(())
}

fn search_chunk_ocl_myers(
    devices: OclRunConfig,
    max_mismatches: u32,
    max_dna_bulges: u32,
    max_rna_bulges: u32,
    search_filter_ascii: &[u8],
    patterns_ascii: &[Vec<u8>],
    n_original_patterns: usize,
    recv: crossbeam_channel::Receiver<SearchChunkInfo>,
    dest: mpsc::SyncSender<SearchChunkResultMyers>,
) -> Result<()> {
    use crate::bit4ops::{char_to_bit4, reverse_compliment_char};
    use crate::myers::build_peq;

    // Precompute shared per-pattern tables (also used by CPU post-process)
    let peqs: Vec<crate::myers::PeqTable> =
        patterns_ascii.iter().map(|p| build_peq(p)).collect();
    let pattern_is_n: Vec<Vec<bool>> = patterns_ascii
        .iter()
        .map(|p| p.iter().map(|&c| c == b'N' || c == b'n').collect())
        .collect();
    let filter_revcomp = reverse_compliment_char(search_filter_ascii);
    let effective_filters_bit4: Vec<Vec<u8>> = patterns_ascii
        .iter()
        .enumerate()
        .map(|(p_idx, _)| {
            let f_ascii: &[u8] = if p_idx < n_original_patterns {
                search_filter_ascii
            } else {
                &filter_revcomp
            };
            f_ascii.iter().map(|&c| char_to_bit4(c)).collect()
        })
        .collect();

    let pam_precheck = precompute_pam_precheck(&pattern_is_n, &effective_filters_bit4);

    // Flatten PAM precheck for GPU buffers
    let pam_len = pam_precheck.iter().map(|(_, f)| f.len()).max().unwrap_or(0);
    let pam_offsets_gpu: Vec<u8> = pam_precheck.iter().map(|(off, _)| *off as u8).collect();
    let mut pam_filters_flat_gpu: Vec<u8> = Vec::with_capacity(patterns_ascii.len() * pam_len);
    for (_, ref filter) in &pam_precheck {
        pam_filters_flat_gpu.extend_from_slice(filter);
        // Pad to pam_len with 0xFF (match anything) if shorter
        for _ in filter.len()..pam_len {
            pam_filters_flat_gpu.push(0xFF);
        }
    }

    let peq_array = Arc::new(build_peq_array(&peqs));
    let pam_off_arc = Arc::new(pam_offsets_gpu);
    let pam_filt_arc = Arc::new(pam_filters_flat_gpu);
    let patterns_arc = Arc::new(patterns_ascii.to_vec());
    let n_patterns = patterns_ascii.len() as u32;
    let n_fwd_patterns = n_original_patterns as u32;

    let pattern_len = patterns_ascii[0].len();
    let pattern_bytes = (pattern_len + 1) / 2;
    // Flatten patterns to bit4-packed form for the kernel.
    let mut pattern_bit4_flat: Vec<u8> = vec![0u8; pattern_bytes * patterns_ascii.len()];
    for (p_idx, pat) in patterns_ascii.iter().enumerate() {
        crate::string_to_bit4(
            &mut pattern_bit4_flat[p_idx * pattern_bytes..(p_idx + 1) * pattern_bytes],
            pat,
            0,
            true,
        );
    }
    let pattern_bit4_arc = Arc::new(pattern_bit4_flat);

    let max_edits = max_mismatches + max_dna_bulges + max_rna_bulges;
    let text_window = pattern_len + max_dna_bulges as usize;
    // Match buffer sized to the search chunk: one candidate slot per 8
    // genome nucleotides. With a typical PAM selectivity of ~1/16 this
    // leaves ~2x headroom before the kernel's silent drop-on-overflow
    // would kick in (we also panic on overflow below).
    let out_buf_size: usize = (search_chunk_nucl() / 8)
        .max(1 << 22)
        .min(1 << 28);
    let compile_defs = format!(
        " -DPATTERN_LEN={} -DPAM_LEN={} -DMAX_EDITS={} -DMAX_MISMATCHES={} -DMAX_DNA_BULGES={} -DMAX_RNA_BULGES={} -DTEXT_WINDOW={} -DOUT_BUF_SIZE={}",
        pattern_len,
        pam_len,
        max_edits,
        max_mismatches,
        max_dna_bulges,
        max_rna_bulges,
        text_window,
        out_buf_size,
    );

    let mut threads: Vec<JoinHandle<Result<()>>> = Vec::new();
    for (_, devs) in devices.get().iter() {
        let plat_devs: Vec<*mut std::ffi::c_void> = devs.iter().map(|d| d.id()).collect();
        if !plat_devs.is_empty() {
            let context = Arc::new(unsafe {
                context::Context::from_devices(&plat_devs, &[0], None, null_mut())?
            });
            let p_devices: Vec<Arc<device::Device>> = plat_devs
                .iter()
                .map(|d| Arc::new(device::Device::new(*d)))
                .collect();
            let program = Arc::new(
                program::Program::create_and_build_from_source(
                    &context,
                    KERNEL_MYERS_CONTENTS,
                    &compile_defs,
                )
                .map_err(|err| {
                    eprintln!("{}", err);
                })
                .unwrap(),
            );

            for p_dev in p_devices {
                let t_dest = dest.clone();
                let t_recv = recv.clone();
                let t_context = context.clone();
                let t_prog = program.clone();
                let t_peq = peq_array.clone();
                let t_pat_b4 = pattern_bit4_arc.clone();
                let t_pam_off = pam_off_arc.clone();
                let t_pam_filt = pam_filt_arc.clone();
                let t_pat = patterns_arc.clone();
                threads.push(thread::spawn(move || unsafe {
                    search_device_ocl_myers(
                        max_mismatches,
                        max_dna_bulges,
                        max_rna_bulges,
                        t_peq,
                        t_pat_b4,
                        t_pam_off,
                        t_pam_filt,
                        n_patterns,
                        n_fwd_patterns,
                        t_pat,
                        t_context,
                        t_prog,
                        p_dev,
                        t_recv,
                        t_dest,
                        out_buf_size,
                    )
                }));
            }
        }
    }
    for t in threads {
        t.join().unwrap()?;
    }
    Ok(())
}

fn convert_matches_myers(
    patterns_ascii: &[Vec<u8>],
    pattern_len: usize,
    max_dna_bulges: u32,
    search_res: SearchChunkResultMyers,
) -> Vec<Match> {
    let n_orig = patterns_ascii.len() / 2;
    let mut results: Vec<Match> = Vec::new();
    let n_chunks = search_res.meta.chr_names.len();
    // Max text extent of a candidate = pattern_len + max_dna_bulges. Needed
    // for the end-of-chromosome check so a bulge match can't leak its tail
    // past the chromosome boundary.
    let max_text_extent = pattern_len as u64 + max_dna_bulges as u64;
    for (smatch, align) in search_res.matches.iter().zip(search_res.alignments.iter()) {
        // chunk_idx is the nucleotide index within the tight-packed search buffer;
        // locate its owning chunk via binary search on chunk_byte_offsets (byte units).
        let nucl_pos = smatch.chunk_idx as usize;
        let byte_pos = nucl_pos / 2;
        let idx = search_res
            .meta
            .chunk_byte_offsets
            .partition_point(|&off| (off as usize) <= byte_pos)
            .saturating_sub(1);
        let chunk_byte_base = search_res.meta.chunk_byte_offsets[idx] as usize;
        let offset = nucl_pos - chunk_byte_base * 2;
        let pos = search_res.meta.chunk_starts[idx] + offset as u64;
        let is_last_chunk = search_res.meta.has_overlap_tail && idx == n_chunks - 1;
        let is_end_chrom = idx == n_chunks - 1
            || search_res.meta.chunk_starts[idx + 1] == 0;
        let is_past_end = pos + max_text_extent > search_res.meta.chunk_ends[idx];
        if is_last_chunk || (is_end_chrom && is_past_end) {
            continue;
        }

        let is_forward = (smatch.pattern_idx as usize) < n_orig;
        // dna_seq / rna_seq are the aligned strings (with '-' gaps)
        let mut dna_result = align.text_aligned.clone();
        let mut rna_result = align.pattern_aligned.clone();
        if !is_forward {
            reverse_compliment_char_i(&mut dna_result);
            reverse_compliment_char_i(&mut rna_result);
        }

        results.push(Match {
            chr_name: search_res.meta.chr_names[idx].clone(),
            chrom_idx: pos,
            pattern_idx: smatch.pattern_idx,
            mismatches: smatch.mismatches,
            is_forward,
            dna_seq: dna_result,
            rna_seq: rna_result,
            dna_bulge_size: smatch.dna_bulge_size as u32,
            rna_bulge_size: smatch.rna_bulge_size as u32,
        });
    }
    results
}

fn chunks_to_searchchunk(
    chunk_buf: &[ChromChunkInfo],
    has_overlap_tail: bool,
    has_overlap_head: bool,
) -> SearchChunkInfo {
    // Tight-pack the buffer: each chunk contributes `ceil(valid_nucl / 2)`
    // bytes, laid out contiguously. This removes the per-chromosome NUL
    // padding that fixed-stride packing would leave behind. chunk_byte_offsets
    // gives the start of each chunk + one sentinel so downstream code can
    // binary-search a genome position back to its chunk.
    let cps = chunks_per_search();
    let max_bytes = search_chunk_bytes();
    let mut search_buf: Box<[u8]> = vec![0_u8; max_bytes].into_boxed_slice();
    let mut names: Vec<String> = Vec::with_capacity(cps);
    let mut starts: Vec<u64> = Vec::with_capacity(cps);
    let mut ends: Vec<u64> = Vec::with_capacity(cps);
    let mut byte_offsets: Vec<u32> = Vec::with_capacity(cps + 1);

    let mut cur_bytes: usize = 0;
    for (idx, chunk) in chunk_buf.iter().enumerate() {
        assert!(
            idx == 0 || *ends.last().unwrap() == chunk.chunk_start || chunk.chunk_start == 0,
            "search expects chromosome chunks to arrive in order"
        );
        let valid_nucl = (chunk.chunk_end - chunk.chunk_start) as usize;
        let valid_bytes = (valid_nucl + 1) / 2;
        assert!(
            cur_bytes + valid_bytes <= max_bytes,
            "tight-packed chunks exceed search_chunk_bytes budget"
        );
        byte_offsets.push(cur_bytes as u32);
        search_buf[cur_bytes..cur_bytes + valid_bytes]
            .copy_from_slice(&chunk.data[..valid_bytes]);
        names.push(chunk.chr_name.clone());
        starts.push(chunk.chunk_start);
        ends.push(chunk.chunk_end);
        cur_bytes += valid_bytes;
    }
    byte_offsets.push(cur_bytes as u32); // sentinel: end of last chunk

    SearchChunkInfo {
        data: search_buf,
        meta: SearchChunkMeta {
            chr_names: names,
            chunk_starts: starts,
            chunk_ends: ends,
            chunk_byte_offsets: byte_offsets,
            has_overlap_tail,
            has_overlap_head,
        },
    }
}
fn convert_matches(
    pattern_len: usize,
    patterns: &Vec<Vec<u8>>,
    search_res: SearchChunkResult,
) -> Vec<Match> {
    let n_blocks = search_res.meta.chr_names.len();
    assert!(n_blocks == search_res.meta.chunk_ends.len());
    assert!(n_blocks == search_res.meta.chunk_starts.len());
    assert!(n_blocks == search_res.meta.chr_names.len());
    let mut results: Vec<Match> = Vec::new();
    for smatch in search_res.matches.iter() {
        let nucl_pos = smatch.chunk_idx as usize;
        let byte_pos = nucl_pos / 2;
        let idx = search_res
            .meta
            .chunk_byte_offsets
            .partition_point(|&off| (off as usize) <= byte_pos)
            .saturating_sub(1);
        let chunk_byte_base = search_res.meta.chunk_byte_offsets[idx] as usize;
        let offset = nucl_pos - chunk_byte_base * 2;
        let pos = search_res.meta.chunk_starts[idx] + offset as u64;
        //skip anything in the last chunk, it will be repeated again in the next search item
        let is_last_chunk = search_res.meta.has_overlap_tail && idx == n_blocks - 1;
        let is_end_chrom = idx == search_res.meta.chr_names.len() - 1
            || search_res.meta.chunk_starts[idx + 1] == 0;
        let is_past_end = pos + pattern_len as u64 > search_res.meta.chunk_ends[idx];
        if !is_last_chunk && !(is_end_chrom && is_past_end) {
            let is_forward = (smatch.pattern_idx as usize) < patterns.len() / 2;
            let mut dna_result: Vec<u8> = vec![0_u8; pattern_len];
            let mut rna_result: Vec<u8> = vec![0_u8; pattern_len];
            bit4_to_string(
                &mut dna_result,
                &search_res.data[..],
                smatch.chunk_idx as usize,
                pattern_len,
            );
            bit4_to_string(
                &mut rna_result,
                &patterns[smatch.pattern_idx as usize][..],
                0,
                pattern_len,
            );
            if !is_forward {
                reverse_compliment_char_i(&mut dna_result);
                reverse_compliment_char_i(&mut rna_result);
            }
            results.push(Match {
                chr_name: search_res.meta.chr_names[idx].clone(),
                chrom_idx: pos,
                pattern_idx: smatch.pattern_idx,
                mismatches: smatch.mismatches,
                is_forward: is_forward,
                dna_seq: dna_result,
                rna_seq: rna_result,
                dna_bulge_size: 0,  // populated by Myers path in later phases
                rna_bulge_size: 0,
            });
        }
    }
    results
}

pub fn search(
    devices: OclRunConfig,
    max_mismatches: u32,
    max_dna_bulges: u32,
    max_rna_bulges: u32,
    search_filter_ascii: &[u8],
    pattern_len: usize,
    patterns_ascii: &Vec<Vec<u8>>,
    recv: mpsc::Receiver<ChromChunkInfo>,
    dest: mpsc::SyncSender<Vec<Match>>,
) {
    /* public facing function, sends and receives data chunk by chunk */
    assert!(!patterns_ascii.is_empty());
    assert!(
        patterns_ascii.iter().all(|p| p.len() == patterns_ascii[0].len()),
        "All patterns must have same length"
    );
    assert!(patterns_ascii[0].len() == pattern_len);

    // Size the search chunk to the GPU's max single-alloc budget (once per
    // process, before any SearchChunkInfo is built). CPU-only runs keep default.
    init_search_chunk_nucl_from_devices(&devices);

    let use_myers = max_dna_bulges > 0 || max_rna_bulges > 0;

    // Convert patterns to bit4 for GPU / legacy CPU path
    let patterns_bit4: Vec<Vec<u8>> = patterns_ascii
        .iter()
        .map(|pat| {
            let mut buf = vec![0_u8; cdiv(pat.len(), 2)];
            crate::string_to_bit4(&mut buf, pat, 0, true);
            buf
        })
        .collect();

    let (compute_send_src, compute_recv_src): (
        crossbeam_channel::Sender<SearchChunkInfo>,
        crossbeam_channel::Receiver<SearchChunkInfo>,
    ) = crossbeam_channel::bounded(4);

    let cps_outer = chunks_per_search();
    let send_thread = thread::Builder::new()
        .stack_size(search_chunk_bytes() * 2)
        .spawn(move || {
            let mut buf: Vec<ChromChunkInfo> = Vec::with_capacity(cps_outer);
            // `has_head` is true once the buffer carries a chunk brought over
            // from a previous batch as pre-overlap (backward context for
            // chunk 1). First batch ever has none. Each batch iterates ALL
            // chunks except the head-overlap one, so there is no tail overlap.
            let mut has_head = false;
            loop {
                let res = recv.recv();
                match res {
                    Ok(chunk) => {
                        buf.push(chunk);
                        if buf.len() == cps_outer {
                            compute_send_src
                                .send(chunks_to_searchchunk(&buf, false, has_head))
                                .unwrap();
                            // Clone the last chunk as pre-overlap for next batch.
                            // Clone is fine because we only need its metadata
                            // and data for backward context, not ownership.
                            let last_el = buf.last().cloned().unwrap();
                            buf.clear();
                            buf.push(last_el);
                            has_head = true;
                        }
                    }
                    Err(_err) => {
                        break;
                    }
                }
            }
            if !buf.is_empty() {
                // Avoid re-emitting the sole pre-overlap chunk as its own batch.
                if !(has_head && buf.len() == 1) {
                    compute_send_src
                        .send(chunks_to_searchchunk(&buf, false, has_head))
                        .unwrap();
                }
            }
        })
        .unwrap();

    if use_myers {
        // Myers path (GPU when devices available, else CPU)
        let (compute_send_dest, compute_recv_dest): (
            mpsc::SyncSender<SearchChunkResultMyers>,
            mpsc::Receiver<SearchChunkResultMyers>,
        ) = mpsc::sync_channel(4);
        let patterns_ascii_clone = patterns_ascii.clone();
        let n_original = patterns_ascii.len() / 2;
        let recv_thread = thread::spawn(move || {
            for search_chunk in compute_recv_dest.iter() {
                dest.send(convert_matches_myers(
                    &patterns_ascii_clone,
                    pattern_len,
                    max_dna_bulges,
                    search_chunk,
                ))
                .unwrap();
            }
        });
        if devices.is_empty() {
            search_compute_cpu_myers(
                max_mismatches,
                max_dna_bulges,
                max_rna_bulges,
                search_filter_ascii,
                patterns_ascii,
                n_original,
                compute_recv_src,
                compute_send_dest,
            );
        } else {
            match search_chunk_ocl_myers(
                devices,
                max_mismatches,
                max_dna_bulges,
                max_rna_bulges,
                search_filter_ascii,
                patterns_ascii,
                n_original,
                compute_recv_src,
                compute_send_dest,
            ) {
                Ok(_) => {}
                Err(err_int) => {
                    panic!("{}", err_int.to_string())
                }
            };
        }
        send_thread.join().unwrap();
        recv_thread.join().unwrap();
    } else {
        // Legacy popcount path (unchanged behavior for bulge=0)
        let (compute_send_dest, compute_recv_dest): (
            mpsc::SyncSender<SearchChunkResult>,
            mpsc::Receiver<SearchChunkResult>,
        ) = mpsc::sync_channel(4);
        let patern_clone = patterns_bit4.clone();
        let recv_thread = thread::spawn(move || {
            for search_chunk in compute_recv_dest.iter() {
                dest.send(convert_matches(pattern_len, &patern_clone, search_chunk))
                    .unwrap();
            }
        });
        if devices.is_empty() {
            search_compute_cpu(
                max_mismatches,
                pattern_len,
                &patterns_bit4,
                compute_recv_src,
                compute_send_dest,
            );
        } else {
            match search_chunk_ocl(
                devices,
                max_mismatches,
                pattern_len,
                &patterns_bit4,
                compute_recv_src,
                compute_send_dest,
            ) {
                Ok(_) => {}
                Err(err_int) => {
                    panic!("{}", err_int.to_string())
                }
            };
        }
        send_thread.join().unwrap();
        recv_thread.join().unwrap();
    }
}

#[cfg(test)]
mod tests {
    use crate::read_2bit;
    use crate::string_to_bit4;
    use std::path::Path;

    // Note this useful idiom: importing names from outer (for mod tests) scope.
    use super::*;

    #[test]
    fn test_opencl_runtime() {
        let platforms = platform::get_platforms().unwrap();
        let num_devices: usize = platforms
            .iter()
            .map(|plat| plat.get_devices(device::CL_DEVICE_TYPE_ALL).unwrap().len())
            .sum();

        assert!(
            num_devices > 0,
            "Needs at least one opencl device to run tests!"
        );
    }
    #[test]
    fn test_search_smoke() {
        let (src_sender, src_receiver): (
            mpsc::SyncSender<ChromChunkInfo>,
            mpsc::Receiver<ChromChunkInfo>,
        ) = mpsc::sync_channel(4);
        let (dest_sender, dest_receiver): (
            mpsc::SyncSender<Vec<Match>>,
            mpsc::Receiver<Vec<Match>>,
        ) = mpsc::sync_channel(4);
        const NUM_ITERS: usize = 2;
        let send_thread = thread::spawn(move || {
            for _ in 0..NUM_ITERS {
                read_2bit(&src_sender, Path::new("tests/test_data/upstream1000.2bit")).unwrap();
            }
        });
        let result_count = thread::spawn(move || {
            let mut count: usize = 0;
            for chunk in dest_receiver.iter() {
                count += chunk.len();
            }
            count
        });
        let pattern1 = b"CCGTGGTTCAACATTTGCTTAGCA".to_vec();
        let pattern2 = b"GATGTTGGTAAGTGGGATATGGCA".to_vec();
        let mut pattern3 = pattern1.clone();
        let mut pattern4 = pattern2.clone();
        pattern3.reverse();
        pattern4.reverse();
        let patterns_ascii: Vec<Vec<u8>> =
            vec![pattern1.clone(), pattern2.clone(), pattern3, pattern4];
        let max_mismatches = 11;
        let expected_results_per_file = 117;
        let expected_results = expected_results_per_file * NUM_ITERS;
        let pattern_len = pattern2.len();
        let search_filter: Vec<u8> = vec![b'N'; pattern_len];
        search(
            OclRunConfig::new(OclDeviceType::CPU).unwrap(),
            max_mismatches,
            0, // max_dna_bulges
            0, // max_rna_bulges
            &search_filter,
            pattern_len,
            &patterns_ascii,
            src_receiver,
            dest_sender,
        );
        send_thread.join().unwrap();
        assert_eq!(result_count.join().unwrap(), expected_results);
    }
}
