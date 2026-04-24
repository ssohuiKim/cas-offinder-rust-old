use crate::bit4ops::{cdiv, roundup};
use crate::run_config::*;
use crate::{bit4_to_string, chrom_chunk::*, reverse_compliment_char_i};
use cudarc::driver::{CudaContext, CudaSlice, DriverError, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::{compile_ptx_with_opts, CompileOptions};
use std::sync::mpsc;
use std::sync::Arc;
use std::thread::{self, JoinHandle};

pub const KERNEL_CONTENTS: &str = include_str!("./kernel.cu");
pub const KERNEL_MYERS_CONTENTS: &str = include_str!("./kernel_myers.cu");

/// Convenience alias so the rest of this module keeps reading like the
/// former OpenCL port (`fn foo(...) -> Result<T>`). CUDA errors all funnel
/// through cudarc's `DriverError`.
pub type Result<T> = std::result::Result<T, DriverError>;

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

/// Set the search chunk size when a CUDA device is available. RTX 4090 /
/// RTX 5090 class GPUs have plenty of VRAM (>= 24 GB), so we just pin the
/// per-batch genome buffer at 1 Gnt (= 512 MB bit4-packed). CPU-only runs
/// keep the default.
pub fn init_search_chunk_nucl_from_devices(devices: &OclRunConfig) {
    if devices.is_empty() {
        return;
    }
    let size = MAX_SEARCH_CHUNK_SIZE;
    SEARCH_CHUNK_NUCL.store(size, std::sync::atomic::Ordering::Relaxed);
}

const CPU_BLOCK_SIZE: usize = 8;
const GPU_BLOCK_SIZE: usize = 4;
const PATTERN_CHUNK_SIZE: usize = 16;
const CL_BLOCKS_PER_EXEC: usize = 4;

#[derive(Clone)]
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
#[derive(Clone, Copy, Debug, Default)]
struct SearchMatch {
    pub chunk_idx: u32,
    pub pattern_idx: u32,
    pub mismatches: u32,
    pub dna_bulge_size: u16,
    pub rna_bulge_size: u16,
}

unsafe impl cudarc::driver::DeviceRepr for SearchMatch {}
unsafe impl cudarc::driver::ValidAsZeroBits for SearchMatch {}

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

// `#[repr(C)]` + no padding issues on the fields we care about → safe to
// hand to cudarc as a POD device element.
unsafe impl cudarc::driver::DeviceRepr for GpuEnumMatch {}
unsafe impl cudarc::driver::ValidAsZeroBits for GpuEnumMatch {}

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
    genome_bit4: &[u8],
    pattern_ascii: &[u8],
) -> Option<(SearchMatch, MyersAlignment)> {
    use crate::bit4ops::{bit4_to_char, get_bit4};
    let ops_count = raw.ops_count as usize;
    let mut pattern_aligned: Vec<u8> = Vec::with_capacity(ops_count);
    let mut text_aligned: Vec<u8> = Vec::with_capacity(ops_count);
    let mut pi: usize = 0;
    let mut gi: usize = raw.chunk_idx as usize;
    for k_rev in (0..ops_count).rev() {
        let op = ((raw.ops_packed >> (k_rev * 2)) & 0x3) as u8;
        match op {
            GPU_OP_MATCH | GPU_OP_SUB => {
                let b4 = get_bit4(genome_bit4, gi);
                if b4 == 0 {
                    return None;
                }
                pattern_aligned.push(pattern_ascii[pi]);
                text_aligned.push(bit4_to_char(b4));
                pi += 1;
                gi += 1;
            }
            GPU_OP_DNA => {
                let b4 = get_bit4(genome_bit4, gi);
                if b4 == 0 {
                    return None;
                }
                pattern_aligned.push(b'-');
                text_aligned.push(bit4_to_char(b4));
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

/// nvrtc JIT doesn't search host include paths by default, so pull the CUDA
/// SDK include directory from whatever env var points at the toolkit root.
/// The .cu kernels need this to resolve `<cstdint>`.
fn nvrtc_sysroot_includes() -> Vec<String> {
    let root = std::env::var("CUDA_PATH")
        .or_else(|_| std::env::var("CUDA_HOME"))
        .or_else(|_| std::env::var("CUDA_ROOT"))
        .unwrap_or_else(|_| "/usr/local/cuda".to_string());
    // conda-forge cuda-toolkit installs headers at $CUDA_PATH/targets/.../include;
    // standard installs also have $CUDA_PATH/include. Feed both.
    vec![
        format!("-I{}/targets/x86_64-linux/include", root),
        format!("-I{}/include", root),
    ]
}

/// Compile-time defines for the legacy popcount kernel. `block_ty` picks the
/// lane width used by the bit-parallel popcount; 32-bit is the efficient
/// choice on NVIDIA GPUs.
fn get_compile_defs(pattern_len: usize) -> Vec<String> {
    let mut out = nvrtc_sysroot_includes();
    out.extend([
        format!("-DPATTERN_LEN={}", pattern_len),
        format!("-DBLOCKS_PER_EXEC={}", CL_BLOCKS_PER_EXEC),
        format!("-DPATTERN_CHUNK_SIZE={}", PATTERN_CHUNK_SIZE),
        "-Dblock_ty=uint32_t".to_string(),
    ]);
    out
}

/// 2D launch config: grid_dim.x covers `n` work items in `BLOCK_X`-sized
/// blocks, grid_dim.y iterates over `m` patterns one per block-y.
fn launch_cfg_2d(n: u32, m: u32) -> LaunchConfig {
    const BLOCK_X: u32 = 128;
    LaunchConfig {
        grid_dim: ((n + BLOCK_X - 1) / BLOCK_X, m.max(1), 1),
        block_dim: (BLOCK_X, 1, 1),
        shared_mem_bytes: 0,
    }
}

fn search_device_cuda(
    max_mismatches: u32,
    pattern_len: usize,
    patterns: Arc<Vec<u8>>,
    ctx: Arc<CudaContext>,
    ptx_source: Arc<String>,
    recv: crossbeam_channel::Receiver<SearchChunkInfo>,
    dest: mpsc::SyncSender<SearchChunkResult>,
) -> Result<()> {
    const OUT_BUF_SIZE: usize = 1 << 22;
    let stream = ctx.default_stream();
    let module = ctx.load_module(cudarc::nvrtc::Ptx::from_src(ptx_source.as_str()))?;
    let kernel = module.load_function("find_matches")?;

    let mut genome_buf: CudaSlice<u8> = stream.alloc_zeros(search_chunk_bytes())?;
    let mut out_count: CudaSlice<i32> = stream.alloc_zeros(1)?;
    let out_buf: CudaSlice<SearchMatch> = stream.alloc_zeros(OUT_BUF_SIZE)?;
    let pattern_buf: CudaSlice<u8> = stream.memcpy_stod(patterns.as_slice())?;

    let pattern_blocked_size = roundup(cdiv(pattern_len, 2), PATTERN_CHUNK_SIZE);
    assert!(patterns.len() % pattern_blocked_size == 0);
    let n_patterns = patterns.len() / pattern_blocked_size;

    for mut item in recv.iter() {
        mask_genome_n_to_zero_inplace(&mut item.data);
        let total_chunks = item.meta.chr_names.len();
        let n_chunks = if item.meta.has_overlap_tail {
            total_chunks - 1
        } else {
            total_chunks
        };
        let n_genome_bytes = item.meta.chunk_byte_offsets[n_chunks] as usize;
        let n_total_bytes = *item.meta.chunk_byte_offsets.last().unwrap() as usize;
        // block_ty = uint32_t → 4 bytes per block.
        let n_genome_blocks = n_genome_bytes / GPU_BLOCK_SIZE;
        let n_genome_execs = n_genome_blocks / CL_BLOCKS_PER_EXEC;
        let match_start_min_nucl: u32 =
            legacy_match_start_min_nucl(&item.meta, pattern_len);

        stream.memcpy_htod(&item.data[..n_total_bytes], &mut genome_buf)?;
        let zero: [i32; 1] = [0];
        stream.memcpy_htod(&zero, &mut out_count)?;

        let n_execs_u32 = n_genome_execs as u32;
        let n_patterns_u32 = n_patterns as u32;
        let cfg = launch_cfg_2d(n_execs_u32, n_patterns_u32);
        let mut launch = stream.launch_builder(&kernel);
        launch.arg(&genome_buf);
        launch.arg(&pattern_buf);
        launch.arg(&max_mismatches);
        launch.arg(&match_start_min_nucl);
        launch.arg(&n_execs_u32);
        launch.arg(&n_patterns_u32);
        launch.arg(&out_buf);
        launch.arg(&out_count);
        unsafe { launch.launch(cfg) }?;

        let count_vec = stream.memcpy_dtov(&out_count)?;
        let readsize = count_vec[0].max(0) as usize;
        if readsize > 0 {
            let read_slice = out_buf.slice(..readsize);
            let outvec = stream.memcpy_dtov(&read_slice)?;
            dest.send(SearchChunkResult {
                matches: outvec,
                meta: item.meta,
                data: item.data,
            })
            .unwrap();
        }
    }
    Ok(())
}

fn search_chunk_cuda(
    devices: OclRunConfig,
    max_mismatches: u32,
    pattern_len: usize,
    patterns: &[Vec<u8>],
    recv: crossbeam_channel::Receiver<SearchChunkInfo>,
    dest: mpsc::SyncSender<SearchChunkResult>,
) -> Result<()> {
    let pattern_arc = Arc::new(pack_patterns(patterns));
    // Compile the .cu source once with runtime nvrtc (pattern_len etc. are
    // baked in as -D defines); every device re-uses the same PTX.
    let defs = get_compile_defs(pattern_len);
    let opts = CompileOptions {
        options: defs,
        ..Default::default()
    };
    let ptx = compile_ptx_with_opts(KERNEL_CONTENTS, opts)
        .expect("nvrtc failed to compile kernel.cu");
    // cudarc's `Ptx::from_src(&str)` takes the PTX text directly, so keep
    // the compiled text around as a shared string.
    let ptx_source = Arc::new(ptx.to_src().to_string());

    let mut threads: Vec<JoinHandle<Result<()>>> = Vec::new();
    for ctx in devices.contexts().iter().cloned() {
        let t_dest = dest.clone();
        let t_recv = recv.clone();
        let t_pattern = pattern_arc.clone();
        let t_ptx = ptx_source.clone();
        threads.push(thread::spawn(move || {
            search_device_cuda(
                max_mismatches,
                pattern_len,
                t_pattern,
                ctx,
                t_ptx,
                t_recv,
                t_dest,
            )
        }));
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
    u64::from_ne_bytes([d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7]])
}
fn block_data_cpu(data: &[u8]) -> Vec<u64> {
    data.chunks(8).map(pack).collect()
}
fn search_chunk_cpu(
    max_mismatches: u32,
    pattern_len: usize,
    packed_patterns: &[u8],
    data: &[u8],
    match_start_min_nucl: u32,
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
                    // Use signed arithmetic: genome 'N' (bit4=0xF) can inflate
                    // popcount above pattern_len at pattern-N positions so the
                    // unsigned subtraction would underflow. Negative values
                    // here still mean "at least as matched as a zero-mm hit",
                    // so they must pass the threshold. main.rs recomputes the
                    // real mismatch count against cas-offinder semantics.
                    let mismatches_i =
                        pattern_len as i32 - num_matches[o] as i32;
                    if mismatches_i <= max_mismatches as i32 {
                        let start_nucl = ((gen_idx + o) * NUCL_PER_BLOCK + l) as u32;
                        // Skip matches whose START lies inside the head-overlap
                        // region AND whose window fits entirely within chunk 0 —
                        // those were already emitted by the previous batch.
                        // Matches that start in chunk 0 but extend into chunk 1
                        // (start_nucl >= match_start_min_nucl) are kept because
                        // the previous batch rejected them via is_past_end.
                        if start_nucl < match_start_min_nucl {
                            continue;
                        }
                        // Cap kernel mm at 0 when negative; main.rs overrides
                        // this with the real count anyway.
                        let mismatches = mismatches_i.max(0) as u32;
                        matches.push(SearchMatch {
                            chunk_idx: start_nucl,
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
    for mut schunk in recv.iter() {
        let match_start_min_nucl = legacy_match_start_min_nucl(&schunk.meta, pattern_len);
        // Popcount semantics treat genome 'N' as a mismatch (cas-offinder C++
        // behaviour). The bit4 genome buffer encodes 'N' as 0xF to distinguish
        // it from chunk-boundary padding (0) — but popcount wants N to zero
        // out so `pattern_bit4 & genome_bit4` doesn't spuriously "match" at
        // ambiguous bases. Mask the copy used by this path.
        mask_genome_n_to_zero_inplace(&mut schunk.data);
        dest.send(SearchChunkResult {
            matches: search_chunk_cpu(
                max_mismatches,
                pattern_len,
                &packed_patterns,
                &schunk.data,
                match_start_min_nucl,
            ),
            meta: schunk.meta,
            data: schunk.data,
        })
        .unwrap();
    }
}

/// Zero out any bit4 nibble equal to 0xF (genome 'N' / wildcard). Used by the
/// legacy popcount path to restore the C++ cas-offinder semantic that genome
/// ambiguity counts as a mismatch at specific-base pattern positions. The
/// mask is applied in place because every chunk is consumed exactly once.
fn mask_genome_n_to_zero_inplace(data: &mut [u8]) {
    // Bit-parallel across u64 lanes: a nibble equals 0xF iff all four of its
    // bits are set, so `x & x>>1 & x>>2 & x>>3` isolates the lowest bit of
    // each 0xF nibble into the `low-bit of each nibble` position.
    const NIBBLE_LOW: u64 = 0x1111_1111_1111_1111;
    let (prefix, mid, suffix) = unsafe { data.align_to_mut::<u64>() };
    for b in prefix.iter_mut().chain(suffix.iter_mut()) {
        let hi = (*b >> 4) & 0x0F;
        let lo = *b & 0x0F;
        let hi_masked = if hi == 0x0F { 0 } else { hi };
        let lo_masked = if lo == 0x0F { 0 } else { lo };
        *b = (hi_masked << 4) | lo_masked;
    }
    for w in mid.iter_mut() {
        let x = *w;
        let all_set = x & (x >> 1) & (x >> 2) & (x >> 3) & NIBBLE_LOW;
        let full_mask = all_set | (all_set << 1) | (all_set << 2) | (all_set << 3);
        *w = x & !full_mask;
    }
}

/// For the legacy popcount path, compute the lowest match START nucleotide
/// that must be emitted in this batch. Matches whose window fits entirely
/// inside chunk 0 (the head-overlap carried from the previous batch) were
/// already emitted there; skip them. Matches starting in chunk 0 but
/// extending into chunk 1 were rejected by the previous batch via
/// `is_past_end` and MUST be emitted here, so the threshold stops
/// `pattern_len - 1` short of chunk 0's end.
fn legacy_match_start_min_nucl(meta: &SearchChunkMeta, pattern_len: usize) -> u32 {
    if !meta.has_overlap_head {
        return 0;
    }
    let chunk1_start_nucl = meta.chunk_byte_offsets[1] as usize * 2;
    chunk1_start_nucl.saturating_sub(pattern_len - 1) as u32
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
}

/// Given a Myers candidate (alignment ending at `end_pos` in the bit4 genome
/// buffer), enumerate every valid traceback. Returns one (SearchMatch,
/// MyersAlignment) per distinct bulge placement / mismatch set that satisfies
/// all per-type caps and the PAM constraint — matching original cas-offinder's
/// behavior of emitting each alternative placement as its own hit.
///
/// Works on bit4 throughout: pattern and text windows are bit4, so the DP and
/// PAM checks compare with `(a & b) != 0` directly (no ASCII round-trip).
#[allow(clippy::too_many_arguments)]
fn classify_myers_candidate(
    end_pos: usize,
    pattern_idx: usize,
    genome_bit4: &[u8],
    patterns_bit4: &[Vec<u8>],
    pattern_is_n: &[Vec<bool>],
    effective_filters_bit4: &[Vec<u8>],
    max_mismatches: u32,
    max_dna_bulges: u32,
    max_rna_bulges: u32,
    text_window_len: usize,
) -> Vec<(SearchMatch, MyersAlignment)> {
    use crate::bit4ops::get_bit4;
    use crate::traceback::{traceback_all, EditOp};

    let max_edits = max_mismatches + max_dna_bulges + max_rna_bulges;
    let pattern_bit4 = &patterns_bit4[pattern_idx];
    let is_n_pat = &pattern_is_n[pattern_idx];
    let eff_filter = &effective_filters_bit4[pattern_idx];

    let text_start_in_chunk = (end_pos + 1).saturating_sub(text_window_len);
    let text_len = end_pos + 1 - text_start_in_chunk;
    let text_bit4: Vec<u8> = (text_start_in_chunk..=end_pos)
        .map(|i| get_bit4(genome_bit4, i))
        .collect();

    let alignments = traceback_all(
        pattern_bit4,
        &text_bit4,
        max_edits,
        max_dna_bulges,
        max_rna_bulges,
        max_mismatches,
        is_n_pat,
    );
    let mut out: Vec<(SearchMatch, MyersAlignment)> = Vec::with_capacity(alignments.len());

    'alignments: for align in alignments {
        // PAM check: for each pattern position that maps to a genome base,
        // verify the effective filter matches. Walk ops forward, pulling bit4
        // directly from the packed genome buffer.
        let genome_span: usize = align.text_aligned.iter().filter(|&&c| c != b'-').count();
        let align_text_offset = text_len - genome_span;
        let mut p_in_pat = 0usize;
        let mut g_off = align_text_offset;
        for op in &align.ops {
            match op {
                EditOp::Match | EditOp::Substitution => {
                    let g_bit4 = get_bit4(genome_bit4, text_start_in_chunk + g_off);
                    let f_bit4 = eff_filter[p_in_pat];
                    // cas-offinder rule: filter N (0xF) admits any genome base
                    // (including genome N); filter specific base rejects padding
                    // (0) and genome N (0xF).
                    let filter_ok = if f_bit4 == 0xF {
                        g_bit4 != 0
                    } else if g_bit4 == 0 || g_bit4 == 0xF {
                        false
                    } else {
                        (g_bit4 & f_bit4) != 0
                    };
                    if !filter_ok {
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
/// For PAM-first patterns (pam_offset == 0) the genome-side PAM position is
/// `end_pos + 1 - (pattern_len + b_dna - b_rna) + pam_offset + k`, so it
/// shifts by `b_rna - b_dna`. We sweep every (b_dna, b_rna) combination and
/// accept if any of them yields a matching PAM. For PAM-last patterns the
/// genome PAM sits at `end_pos - pam_len + 1 + k` regardless of bulges (RNA
/// bulges may not lie in the PAM region, and DNA bulges before the PAM don't
/// move the anchor from `end_pos`) so a single check is enough.
fn check_pam_quick(
    genome_bit4: &[u8],
    total_nucl: usize,
    end_pos: usize,
    pattern_len: usize,
    pam_offset: usize,
    pam_filter: &[u8],
    max_dna_bulges: u32,
    max_rna_bulges: u32,
) -> bool {
    let (db_range, rb_range) = if pam_offset == 0 {
        (max_dna_bulges as i64 + 1, max_rna_bulges as i64 + 1)
    } else {
        (1i64, 1i64)
    };
    for b_dna in 0..db_range {
        for b_rna in 0..rb_range {
            let genome_span = pattern_len as i64 + b_dna - b_rna;
            if genome_span <= 0 || genome_span > (end_pos as i64 + 1) {
                continue;
            }
            let align_start = (end_pos + 1) - genome_span as usize;
            let mut ok = true;
            for (k, &f) in pam_filter.iter().enumerate() {
                let gpos = align_start + pam_offset + k;
                if gpos >= total_nucl {
                    ok = false;
                    break;
                }
                let g = crate::bit4ops::get_bit4(genome_bit4, gpos);
                // cas-offinder PAM rule: filter N (0xF) accepts any genome
                // base including genome N; a specific filter base must match
                // the genome base, and genome N does NOT satisfy a specific
                // filter position.
                let pos_ok = if f == 0xF {
                    g != 0 // still reject padding
                } else if g == 0 || g == 0xF {
                    false
                } else {
                    (g & f) != 0
                };
                if !pos_ok {
                    ok = false;
                    break;
                }
            }
            if ok {
                return true;
            }
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
    patterns_bit4: &[Vec<u8>],
    pattern_len: usize,
    effective_filters_bit4: &[Vec<u8>],
    pattern_is_n: &[Vec<bool>],
    pam_precheck: &[(usize, Vec<u8>)],
    peqs: &[crate::myers::PeqTable],
    chunk_data_bit4: &[u8],
    active_start_nucl: usize,
    active_end_nucl: usize,
) -> (Vec<SearchMatch>, Vec<MyersAlignment>) {
    use crate::bit4ops::get_bit4;

    let max_edits = max_mismatches + max_dna_bulges + max_rna_bulges;
    let total_nucl = active_end_nucl.min(chunk_data_bit4.len() * 2);
    let text_window_len = pattern_len + max_dna_bulges as usize;

    let mut matches: Vec<SearchMatch> = Vec::new();
    let mut alignments: Vec<MyersAlignment> = Vec::new();

    let mask = if pattern_len < 64 {
        (1u64 << pattern_len) - 1
    } else {
        !0u64
    };
    let last_bit = 1u64 << (pattern_len - 1);

    // Per-position windowed Myers on BIT4 end-to-end — no ASCII round-trip.
    // PAM pre-check, Myers sweep, and classify all read nibbles directly from
    // the packed genome buffer via `get_bit4`. This matches the GPU kernel
    // byte-for-byte so CPU/GPU enumerate the same candidate set.
    for (p_idx, peq) in peqs.iter().enumerate() {
        let (pam_offset, ref pam_filter) = pam_precheck[p_idx];

        for t_pos in active_start_nucl..total_nucl {
            if t_pos + 1 < pattern_len {
                continue;
            }

            // PAM pre-check first (cheap fail-fast, same order as kernel)
            if !check_pam_quick(
                chunk_data_bit4,
                total_nucl,
                t_pos,
                pattern_len,
                pam_offset,
                pam_filter,
                max_dna_bulges,
                max_rna_bulges,
            ) {
                continue;
            }

            // Fresh Myers sweep over the candidate's text window, mirroring
            // the GPU kernel exactly.
            //   bit4 == 0    → padding (no peq bits contribute)
            //   bit4 == 0xF  → genome 'N' — per cas-offinder, match only
            //                  positions where pattern is 'N'
            //   otherwise    → OR the per-base peqs matching set bits
            let text_start = (t_pos + 1).saturating_sub(text_window_len);
            let mut vp: u64 = mask;
            let mut vn: u64 = 0;
            let mut score: i32 = pattern_len as i32;
            for t in text_start..=t_pos {
                let b4 = get_bit4(chunk_data_bit4, t);
                let eq: u64 = if b4 == 0 {
                    0
                } else if b4 == 0xF {
                    peq.peq_n
                } else {
                    let mut e = 0u64;
                    if b4 & 0x4 != 0 { e |= peq.peq[0]; } // A
                    if b4 & 0x2 != 0 { e |= peq.peq[1]; } // C
                    if b4 & 0x8 != 0 { e |= peq.peq[2]; } // G
                    if b4 & 0x1 != 0 { e |= peq.peq[3]; } // T
                    e
                };
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
                chunk_data_bit4,
                patterns_bit4,
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
    patterns_bit4: Arc<Vec<Vec<u8>>>,
    pattern_len: usize,
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
            &patterns_bit4,
            pattern_len,
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

    // Per-character bit4 patterns (one nibble per nucleotide, length =
    // pattern_len). The CPU DP + PAM verification read these directly,
    // avoiding any ASCII round-trip.
    let pattern_len = patterns_ascii[0].len();
    let patterns_bit4: Vec<Vec<u8>> = patterns_ascii
        .iter()
        .map(|p| p.iter().map(|&c| char_to_bit4(c)).collect())
        .collect();

    let peqs_arc = Arc::new(peqs);
    let is_n_arc = Arc::new(pattern_is_n);
    let filters_arc = Arc::new(effective_filters_bit4);
    let pam_arc = Arc::new(pam_precheck);
    let patterns_b4_arc = Arc::new(patterns_bit4);

    let n_threads: usize = thread::available_parallelism().unwrap().into();
    let mut threads: Vec<thread::JoinHandle<()>> = Vec::new();
    for _ in 0..n_threads {
        let t_pat = patterns_b4_arc.clone();
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
                pattern_len,
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

/// Layout per pattern: [A, C, G, T, N] — 5 u64 words. Slot 4 is `peq_n`, the
/// bit mask of pattern positions holding 'N' (wildcard), used when the genome
/// position is itself 'N' (bit4==0xF). Must match the indexing in
/// kernel_myers.cl (PEQ_STRIDE = 5).
fn build_peq_array(peqs: &[crate::myers::PeqTable]) -> Vec<u64> {
    let mut out = Vec::with_capacity(peqs.len() * 5);
    for peq in peqs {
        out.push(peq.peq[0]);
        out.push(peq.peq[1]);
        out.push(peq.peq[2]);
        out.push(peq.peq[3]);
        out.push(peq.peq_n);
    }
    out
}

#[allow(clippy::too_many_arguments)]
fn search_device_cuda_myers(
    peq_array: Arc<Vec<u64>>,
    pattern_bit4_flat: Arc<Vec<u8>>,
    pam_offsets_gpu: Arc<Vec<u8>>,
    pam_filters_flat_gpu: Arc<Vec<u8>>,
    n_patterns: u32,
    n_fwd_patterns: u32,
    patterns_ascii: Arc<Vec<Vec<u8>>>,
    ctx: Arc<CudaContext>,
    ptx_source: Arc<String>,
    recv: crossbeam_channel::Receiver<SearchChunkInfo>,
    dest: mpsc::SyncSender<SearchChunkResultMyers>,
    out_buf_size: usize,
) -> Result<()> {
    let stream = ctx.default_stream();
    let module = ctx.load_module(cudarc::nvrtc::Ptx::from_src(ptx_source.as_str()))?;
    let kernel = module.load_function("find_matches_myers")?;

    let mut genome_buf: CudaSlice<u8> = stream.alloc_zeros(search_chunk_bytes())?;
    let mut out_count: CudaSlice<u32> = stream.alloc_zeros(1)?;
    let out_buf: CudaSlice<GpuEnumMatch> = stream.alloc_zeros(out_buf_size)?;
    let peq_buf: CudaSlice<u64> = stream.memcpy_stod(peq_array.as_slice())?;
    let pattern_buf: CudaSlice<u8> = stream.memcpy_stod(pattern_bit4_flat.as_slice())?;
    let pam_off_buf: CudaSlice<u8> = stream.memcpy_stod(pam_offsets_gpu.as_slice())?;
    let pam_filt_buf: CudaSlice<u8> = stream.memcpy_stod(pam_filters_flat_gpu.as_slice())?;

    for item in recv.iter() {
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

        stream.memcpy_htod(&item.data[..n_write_bytes], &mut genome_buf)?;

        // Launch the kernel over a shrinking stack of sub-ranges. Each
        // sub-range runs independently against `out_buf`; if the atomic
        // write counter exceeds `out_buf_size` we know the kernel dropped
        // matches past that index, so we bisect the range and re-enqueue
        // the two halves. Sub-ranges that fit emit their matches straight
        // into the streaming pipeline — `raw_matches`, the decoded Vecs
        // and the produced `Match` list are bounded by `out_buf_size`,
        // not by the whole-batch candidate count, so peak memory stays
        // roughly constant regardless of db/rb/PAM selectivity.
        let mut total_emitted = 0usize;
        let mut pending: Vec<(u32, u32)> = vec![(active_start_nucl, n_active_nucl)];
        while let Some((sub_start, sub_end)) = pending.pop() {
            if sub_start >= sub_end {
                continue;
            }
            let zero: [u32; 1] = [0];
            stream.memcpy_htod(&zero, &mut out_count)?;

            let sub_len = sub_end - sub_start;
            // Kernel's `j_start` parameter replaces OpenCL's
            // clEnqueueNDRangeKernel global_work_offset: host launches with
            // grid sized for the sub-range width; kernel adds `j_start` to
            // its thread index to recover the absolute `j`.
            let cfg = launch_cfg_2d(sub_len, n_patterns);
            let mut launch = stream.launch_builder(&kernel);
            launch.arg(&genome_buf);
            launch.arg(&peq_buf);
            launch.arg(&pattern_buf);
            launch.arg(&pam_off_buf);
            launch.arg(&pam_filt_buf);
            launch.arg(&n_patterns);
            launch.arg(&n_fwd_patterns);
            launch.arg(&sub_start);
            launch.arg(&sub_end);
            launch.arg(&active_start_nucl);
            launch.arg(&n_active_nucl);
            launch.arg(&out_buf);
            launch.arg(&out_count);
            unsafe { launch.launch(cfg) }?;

            let count_vec = stream.memcpy_dtov(&out_count)?;
            let total_found = count_vec[0] as usize;

            if total_found > out_buf_size {
                let mid = sub_start + (sub_end - sub_start) / 2;
                if mid == sub_start || mid == sub_end {
                    panic!(
                        "GPU match buffer overflow on minimal sub-range \
                         [{sub_start}, {sub_end}): {} candidates > {}. \
                         Increase out_buf_size.",
                        total_found, out_buf_size
                    );
                }
                pending.push((mid, sub_end));
                pending.push((sub_start, mid));
                continue;
            }
            if total_found == 0 {
                continue;
            }

            let read_slice = out_buf.slice(..total_found);
            let raw_matches: Vec<GpuEnumMatch> = stream.memcpy_dtov(&read_slice)?;

            use rayon::prelude::*;
            let decoded: Vec<(SearchMatch, MyersAlignment)> = raw_matches
                .par_iter()
                .filter_map(|raw| {
                    let pattern_ascii = &patterns_ascii[raw.pattern_idx as usize];
                    decode_gpu_enum_match(raw, &item.data[..], pattern_ascii)
                })
                .collect();
            drop(raw_matches);
            let (final_matches, alignments): (Vec<_>, Vec<_>) = decoded.into_iter().unzip();
            total_emitted += final_matches.len();

            dest.send(SearchChunkResultMyers {
                matches: final_matches,
                alignments,
                meta: item.meta.clone(),
            })
            .unwrap();
        }

        if total_emitted == 0 {
            dest.send(SearchChunkResultMyers {
                matches: Vec::new(),
                alignments: Vec::new(),
                meta: item.meta,
            })
            .unwrap();
        }
    }
    Ok(())
}

fn search_chunk_cuda_myers(
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

    let pam_len = pam_precheck.iter().map(|(_, f)| f.len()).max().unwrap_or(0);
    let pam_offsets_gpu: Vec<u8> = pam_precheck.iter().map(|(off, _)| *off as u8).collect();
    let mut pam_filters_flat_gpu: Vec<u8> = Vec::with_capacity(patterns_ascii.len() * pam_len);
    for (_, ref filter) in &pam_precheck {
        pam_filters_flat_gpu.extend_from_slice(filter);
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
    // Fixed 4 M-slot output buffer; sub-range bisect handles overflow (see
    // search_device_cuda_myers).
    let out_buf_size: usize = 4 * 1024 * 1024;
    let mut opts_vec = nvrtc_sysroot_includes();
    opts_vec.extend([
        format!("-DPATTERN_LEN={}", pattern_len),
        format!("-DPAM_LEN={}", pam_len),
        format!("-DMAX_EDITS={}", max_edits),
        format!("-DMAX_MISMATCHES={}", max_mismatches),
        format!("-DMAX_DNA_BULGES={}", max_dna_bulges),
        format!("-DMAX_RNA_BULGES={}", max_rna_bulges),
        format!("-DTEXT_WINDOW={}", text_window),
        format!("-DOUT_BUF_SIZE={}", out_buf_size),
    ]);
    let compile_opts = CompileOptions {
        options: opts_vec,
        ..Default::default()
    };
    let ptx = compile_ptx_with_opts(KERNEL_MYERS_CONTENTS, compile_opts)
        .expect("nvrtc failed to compile kernel_myers.cu");
    let ptx_source = Arc::new(ptx.to_src());

    let mut threads: Vec<JoinHandle<Result<()>>> = Vec::new();
    for ctx in devices.contexts().iter().cloned() {
        let t_dest = dest.clone();
        let t_recv = recv.clone();
        let t_peq = peq_array.clone();
        let t_pat_b4 = pattern_bit4_arc.clone();
        let t_pam_off = pam_off_arc.clone();
        let t_pam_filt = pam_filt_arc.clone();
        let t_pat = patterns_arc.clone();
        let t_ptx = ptx_source.clone();
        threads.push(thread::spawn(move || {
            search_device_cuda_myers(
                t_peq,
                t_pat_b4,
                t_pam_off,
                t_pam_filt,
                n_patterns,
                n_fwd_patterns,
                t_pat,
                ctx,
                t_ptx,
                t_recv,
                t_dest,
                out_buf_size,
            )
        }));
    }
    for t in threads {
        t.join().unwrap()?;
    }
    Ok(())
}

fn convert_matches_myers(
    patterns_ascii: &[Vec<u8>],
    pattern_len: usize,
    _max_dna_bulges: u32,
    search_res: SearchChunkResultMyers,
) -> Vec<Match> {
    let n_orig = patterns_ascii.len() / 2;
    let mut results: Vec<Match> = Vec::new();
    let n_chunks = search_res.meta.chr_names.len();
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
        // Use this match's actual genome span (pattern_len + dna_bulges - rna_bulges)
        // instead of the worst-case upper bound — otherwise short alignments near
        // a chromosome tail get rejected even when they fit entirely within it.
        let actual_extent = (pattern_len as u64)
            + (smatch.dna_bulge_size as u64)
            - (smatch.rna_bulge_size as u64);
        let is_past_end = pos + actual_extent > search_res.meta.chunk_ends[idx];
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
            match search_chunk_cuda_myers(
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
                    panic!("{:?}", err_int)
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
            match search_chunk_cuda(
                devices,
                max_mismatches,
                pattern_len,
                &patterns_bit4,
                compute_recv_src,
                compute_send_dest,
            ) {
                Ok(_) => {}
                Err(err_int) => {
                    panic!("{:?}", err_int)
                }
            };
        }
        send_thread.join().unwrap();
        recv_thread.join().unwrap();
    }
}

