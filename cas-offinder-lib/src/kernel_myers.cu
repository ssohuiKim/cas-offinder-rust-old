// CUDA port of kernel_myers.cl.
//
// Translation rules applied mechanically from the OpenCL original:
//   __kernel          → extern "C" __global__
//   __global <T>*     → <T>*        (global memory is the default in CUDA)
//   uchar/ushort/uint/ulong → uint8_t/uint16_t/uint32_t/uint64_t
//   get_global_id(0)  → blockIdx.x * blockDim.x + threadIdx.x + j_start
//   get_global_id(1)  → blockIdx.y * blockDim.y + threadIdx.y
//   atomic_inc(ptr)   → atomicAdd(ptr, 1)   (same "return old value" semantics)
//   min(a, b)         → integer_min() below (avoids pulling in <algorithm>)
//
// One extra argument vs the OpenCL version: `j_start`. OpenCL's
// clEnqueueNDRangeKernel takes a per-kernel global work offset; CUDA has no
// equivalent, so host code launches grids sized for the sub-range and passes
// the offset explicitly.
//
// Compile-time defines injected by build.rs/nvcc:
//   PATTERN_LEN, PAM_LEN, MAX_EDITS, MAX_MISMATCHES, MAX_DNA_BULGES,
//   MAX_RNA_BULGES, TEXT_WINDOW, OUT_BUF_SIZE

// nvrtc JIT doesn't have access to host <stdint.h> / <cstdint>, so declare
// the fixed-width ints we use locally.
typedef unsigned char      uint8_t;
typedef unsigned short     uint16_t;
typedef unsigned int       uint32_t;
typedef unsigned long long uint64_t;

#ifndef PATTERN_LEN
#define PATTERN_LEN 27
#endif
#ifndef PAM_LEN
#define PAM_LEN 3
#endif
#ifndef MAX_EDITS
#define MAX_EDITS 11
#endif
#ifndef MAX_MISMATCHES
#define MAX_MISMATCHES 9
#endif
#ifndef MAX_DNA_BULGES
#define MAX_DNA_BULGES 1
#endif
#ifndef MAX_RNA_BULGES
#define MAX_RNA_BULGES 1
#endif
#ifndef TEXT_WINDOW
#define TEXT_WINDOW (PATTERN_LEN + MAX_DNA_BULGES)
#endif
#ifndef OUT_BUF_SIZE
#define OUT_BUF_SIZE (4 * 1024 * 1024)
#endif

#define OP_MATCH 0
#define OP_SUB   1
#define OP_DNA   2
#define OP_RNA   3
#define INF_COST 100
#define PATTERN_BYTES (((PATTERN_LEN) + 1) / 2)
#define MAX_OPS (PATTERN_LEN + MAX_DNA_BULGES)
#define PEQ_STRIDE 5

// cas-offinder match/mismatch rule (bit4):
//   tb4 == 0    → padding, mismatch
//   pb4 == 0xF  → pattern N wildcard, match
//   tb4 == 0xF  → genome N, matches only pattern N
//   otherwise   → (pb4 & tb4) != 0
__device__ __forceinline__ uint8_t match_cost_bit4(uint8_t pb4, uint8_t tb4) {
    if (tb4 == 0)    return 1;
    if (pb4 == 0x0F) return 0;
    if (tb4 == 0x0F) return 1;
    return ((pb4 & tb4) != 0) ? 0 : 1;
}

// PAM-filter rule: filter N accepts any genome base except padding; filter
// specific requires a specific genome base (rejects padding and genome N).
__device__ __forceinline__ uint8_t pam_cell_ok(uint8_t g, uint8_t f) {
    if (f == 0x0F) return (g != 0) ? 1 : 0;
    if (g == 0 || g == 0x0F) return 0;
    return ((g & f) != 0) ? 1 : 0;
}

__device__ __forceinline__ uint32_t integer_min(uint32_t a, uint32_t b) {
    return a < b ? a : b;
}

struct s_match {
    uint32_t chunk_idx;
    uint32_t pattern_idx;
    uint64_t ops_packed;
    uint32_t mismatches;
    uint16_t dna_bulge_size;
    uint16_t rna_bulge_size;
    uint32_t ops_count;
    uint32_t _pad;
};

__device__ __forceinline__ uint8_t get_bit4(const uint8_t* chunk, uint32_t i) {
    uint8_t byte = chunk[i / 2];
    return (i % 2 == 0) ? (byte & 0x0F) : ((byte >> 4) & 0x0F);
}

__device__ __forceinline__ uint8_t get_pattern(
    const uint8_t* patterns, uint32_t p, uint32_t i
) {
    uint32_t base = p * PATTERN_BYTES;
    uint8_t byte = patterns[base + i / 2];
    return (i % 2 == 0) ? (byte & 0x0F) : ((byte >> 4) & 0x0F);
}

extern "C" __global__ void find_matches_myers(
    const uint8_t*  __restrict__ genome_bit4,
    const uint64_t* __restrict__ peq_tables,
    const uint8_t*  __restrict__ pattern_bit4,
    const uint8_t*  __restrict__ pam_offsets,
    const uint8_t*  __restrict__ pam_filters_bit4,
    uint32_t n_patterns,
    uint32_t n_fwd_patterns,
    uint32_t j_start,                // sub-range offset (replaces OpenCL global_work_offset)
    uint32_t active_start_nucl,
    uint32_t total_nucl,
    s_match* __restrict__ out_matches,
    uint32_t* __restrict__ out_count
) {
    uint32_t j = blockIdx.x * blockDim.x + threadIdx.x + j_start;
    uint32_t p = blockIdx.y * blockDim.y + threadIdx.y;

    if (p >= n_patterns) return;
    if (j >= total_nucl) return;
    if (j < active_start_nucl) return;
    if (j + 1 < PATTERN_LEN) return;

    // ---- PAM pre-check. For PAM-first patterns the genome PAM shifts by
    // b_rna - b_dna, so sweep every (b_dna, b_rna) pair. PAM-last has the
    // PAM anchored to j regardless of bulges, so a single (0, 0) suffices.
    uint32_t pam_off = (uint32_t)pam_offsets[p];
    uint32_t db_range = (pam_off == 0) ? (MAX_DNA_BULGES + 1u) : 1u;
    uint32_t rb_range = (pam_off == 0) ? (MAX_RNA_BULGES + 1u) : 1u;
    bool pam_ok = false;
    for (uint32_t b_dna = 0; b_dna < db_range && !pam_ok; b_dna++) {
        for (uint32_t b_rna = 0; b_rna < rb_range; b_rna++) {
            int genome_span = (int)PATTERN_LEN + (int)b_dna - (int)b_rna;
            if (genome_span <= 0) continue;
            if ((int)(j + 1) < genome_span) continue;
            uint32_t align_start_try = (j + 1) - (uint32_t)genome_span;
            bool this_ok = true;
            for (uint32_t k = 0; k < PAM_LEN; k++) {
                uint32_t gpos = align_start_try + pam_off + k;
                if (gpos >= total_nucl) { this_ok = false; break; }
                uint8_t g = get_bit4(genome_bit4, gpos);
                uint8_t f = pam_filters_bit4[p * PAM_LEN + k];
                if (!pam_cell_ok(g, f)) { this_ok = false; break; }
            }
            if (this_ok) { pam_ok = true; break; }
        }
    }
    if (!pam_ok) return;

    // ---- Myers sweep over the text window.
    uint64_t peq_a = peq_tables[p * PEQ_STRIDE + 0];
    uint64_t peq_c = peq_tables[p * PEQ_STRIDE + 1];
    uint64_t peq_g = peq_tables[p * PEQ_STRIDE + 2];
    uint64_t peq_t = peq_tables[p * PEQ_STRIDE + 3];
    uint64_t peq_n = peq_tables[p * PEQ_STRIDE + 4];

    uint64_t mask =
        (PATTERN_LEN < 64) ? (((uint64_t)1 << PATTERN_LEN) - 1) : ~(uint64_t)0;
    uint64_t last_bit = (uint64_t)1 << (PATTERN_LEN - 1);
    uint64_t vp = mask;
    uint64_t vn = 0;
    int score = PATTERN_LEN;

    uint32_t text_start =
        (j + 1 > (uint32_t)TEXT_WINDOW) ? (j + 1 - (uint32_t)TEXT_WINDOW) : 0;
    uint32_t text_len = j + 1 - text_start;
    for (uint32_t t = text_start; t <= j; t++) {
        uint8_t b4 = get_bit4(genome_bit4, t);
        uint64_t eq;
        if (b4 == 0) {
            eq = 0;
        } else if (b4 == 0x0F) {
            eq = peq_n;
        } else {
            eq = 0;
            if (b4 & 0x4) eq |= peq_a;
            if (b4 & 0x2) eq |= peq_c;
            if (b4 & 0x8) eq |= peq_g;
            if (b4 & 0x1) eq |= peq_t;
        }

        uint64_t x = eq | vn;
        uint64_t d0 = (((x & vp) + vp) ^ vp) | x;
        uint64_t hn = vp & d0;
        uint64_t hp = vn | ~(vp | d0);
        uint64_t x_shift = hp << 1;
        vn = x_shift & d0;
        vp = (hn << 1) | ~(x_shift | d0);
        vp &= mask;
        vn &= mask;
        if (hp & last_bit) score++;
        if (hn & last_bit) score--;
    }
    if (score < 0 || score > (int)MAX_EDITS) return;

    // ---- Full DP in registers/local memory.
    uint8_t dna_cost = (MAX_DNA_BULGES == 0) ? INF_COST : 1;
    uint8_t rna_cost = (MAX_RNA_BULGES == 0) ? INF_COST : 1;

    uint8_t dp[(PATTERN_LEN + 1) * (TEXT_WINDOW + 1)];

    for (uint32_t i = 0; i <= PATTERN_LEN; i++) {
        uint32_t v = i * (uint32_t)rna_cost;
        dp[i * (TEXT_WINDOW + 1)] =
            (v > INF_COST) ? (uint8_t)INF_COST : (uint8_t)v;
    }
    for (uint32_t t = 0; t <= TEXT_WINDOW; t++) {
        dp[t] = 0;
    }
    for (uint32_t pi = 1; pi <= PATTERN_LEN; pi++) {
        uint8_t pb4 = get_pattern(pattern_bit4, p, pi - 1);
        for (uint32_t tj = 1; tj <= TEXT_WINDOW; tj++) {
            if (tj > text_len) {
                dp[pi * (TEXT_WINDOW + 1) + tj] = INF_COST;
                continue;
            }
            uint8_t tb4 = get_bit4(genome_bit4, text_start + tj - 1);
            if (tb4 == 0) {
                dp[pi * (TEXT_WINDOW + 1) + tj] = INF_COST;
                continue;
            }
            uint8_t mc = match_cost_bit4(pb4, tb4);
            uint32_t diag =
                (uint32_t)dp[(pi - 1) * (TEXT_WINDOW + 1) + (tj - 1)] + mc;
            uint32_t up =
                (uint32_t)dp[(pi - 1) * (TEXT_WINDOW + 1) + tj] + (uint32_t)rna_cost;
            uint32_t left =
                (uint32_t)dp[pi * (TEXT_WINDOW + 1) + (tj - 1)] + (uint32_t)dna_cost;
            uint32_t v = integer_min(diag, integer_min(up, left));
            if (v > INF_COST) v = INF_COST;
            dp[pi * (TEXT_WINDOW + 1) + tj] = (uint8_t)v;
        }
    }

    uint32_t last_cell = PATTERN_LEN * (TEXT_WINDOW + 1) + text_len;
    uint8_t final_cost = dp[last_cell];
    if (final_cost > (uint8_t)MAX_EDITS) return;

    // ---- Multi-path iterative DFS traceback (same as OpenCL version).
    struct frame {
        uint8_t pi;
        uint8_t tj;
        uint8_t trans;
        uint8_t prev_ops_len;
        uint8_t prev_cost;
        uint8_t prev_mm;
        uint8_t prev_db;
        uint8_t prev_rb;
    };
    frame stack[PATTERN_LEN + MAX_DNA_BULGES + 1];
    uint8_t ops[MAX_OPS];
    uint32_t sp = 0;
    uint32_t ops_len = 0;
    uint8_t cost = 0, mm = 0, db = 0, rb = 0;

    stack[0].pi = PATTERN_LEN;
    stack[0].tj = (uint8_t)text_len;
    stack[0].trans = 0;
    stack[0].prev_ops_len = 0;
    stack[0].prev_cost = 0;
    stack[0].prev_mm = 0;
    stack[0].prev_db = 0;
    stack[0].prev_rb = 0;
    sp = 1;

    while (sp > 0) {
        uint32_t top = sp - 1;
        uint8_t cpi = stack[top].pi;
        uint8_t ctj = stack[top].tj;

        if (cpi == 0) {
            uint32_t genome_span = (uint32_t)PATTERN_LEN + (uint32_t)db - (uint32_t)rb;
            uint32_t match_start = j + 1 - genome_span;

            uint32_t pi_fwd = 0;
            uint32_t g_off = 0;
            bool pam_ok_full = true;
            for (int oi = (int)ops_len - 1; oi >= 0; oi--) {
                uint8_t op = ops[oi];
                if (op == OP_MATCH || op == OP_SUB) {
                    uint8_t pb4 = get_pattern(pattern_bit4, p, pi_fwd);
                    if (pb4 == 0x0F) {
                        uint8_t gb4 = get_bit4(genome_bit4, match_start + g_off);
                        uint32_t k = pi_fwd - pam_off;
                        uint8_t f = pam_filters_bit4[p * PAM_LEN + k];
                        if ((gb4 & f) == 0) { pam_ok_full = false; break; }
                    }
                    pi_fwd++; g_off++;
                } else if (op == OP_DNA) {
                    g_off++;
                } else {
                    pi_fwd++;
                }
            }

            if (pam_ok_full) {
                uint64_t ops_packed = 0;
                for (uint32_t k = 0; k < ops_len; k++) {
                    ops_packed |= ((uint64_t)ops[k]) << (k * 2);
                }
                uint32_t idx = atomicAdd(out_count, 1u);
                if (idx < OUT_BUF_SIZE) {
                    out_matches[idx].chunk_idx = match_start;
                    out_matches[idx].pattern_idx = p;
                    out_matches[idx].ops_packed = ops_packed;
                    out_matches[idx].mismatches = mm;
                    out_matches[idx].dna_bulge_size = (uint16_t)db;
                    out_matches[idx].rna_bulge_size = (uint16_t)rb;
                    out_matches[idx].ops_count = ops_len;
                    out_matches[idx]._pad = 0;
                }
            }

            ops_len = stack[top].prev_ops_len;
            cost = stack[top].prev_cost;
            mm = stack[top].prev_mm;
            db = stack[top].prev_db;
            rb = stack[top].prev_rb;
            sp--;
            continue;
        }

        uint8_t trans = stack[top].trans;

        if (trans == 0) {
            stack[top].trans = 1;
            if (ctj > 0) {
                uint8_t pb4 = get_pattern(pattern_bit4, p, cpi - 1);
                uint8_t tb4 = get_bit4(genome_bit4, text_start + ctj - 1);
                uint8_t mc = match_cost_bit4(pb4, tb4);
                if (mc == 0 || pb4 != 0x0F) {
                    uint8_t new_cost = cost + mc;
                    uint8_t new_mm = mm + mc;
                    uint8_t rem = dp[(cpi - 1) * (TEXT_WINDOW + 1) + (ctj - 1)];
                    if ((uint32_t)new_cost + (uint32_t)rem <= (uint32_t)MAX_EDITS
                        && new_mm <= MAX_MISMATCHES
                        && ops_len < MAX_OPS) {
                        stack[sp].pi = cpi - 1;
                        stack[sp].tj = ctj - 1;
                        stack[sp].trans = 0;
                        stack[sp].prev_ops_len = (uint8_t)ops_len;
                        stack[sp].prev_cost = cost;
                        stack[sp].prev_mm = mm;
                        stack[sp].prev_db = db;
                        stack[sp].prev_rb = rb;
                        ops[ops_len++] = (mc == 0) ? OP_MATCH : OP_SUB;
                        cost = new_cost;
                        mm = new_mm;
                        sp++;
                    }
                }
            }
            continue;
        }

        if (trans == 1) {
            stack[top].trans = 2;
            if (rna_cost != INF_COST && rb < MAX_RNA_BULGES) {
                uint8_t pb4 = get_pattern(pattern_bit4, p, cpi - 1);
                uint8_t next_is_pam = (cpi < PATTERN_LEN)
                    ? (get_pattern(pattern_bit4, p, cpi) == 0x0F) : 0;
                uint8_t prev_is_pam = (cpi >= 2)
                    ? (get_pattern(pattern_bit4, p, cpi - 2) == 0x0F) : 0;
                if (pb4 != 0x0F && !next_is_pam && !prev_is_pam) {
                    uint8_t new_cost = cost + 1;
                    uint8_t rem = dp[(cpi - 1) * (TEXT_WINDOW + 1) + ctj];
                    if ((uint32_t)new_cost + (uint32_t)rem <= (uint32_t)MAX_EDITS
                        && ops_len < MAX_OPS) {
                        stack[sp].pi = cpi - 1;
                        stack[sp].tj = ctj;
                        stack[sp].trans = 0;
                        stack[sp].prev_ops_len = (uint8_t)ops_len;
                        stack[sp].prev_cost = cost;
                        stack[sp].prev_mm = mm;
                        stack[sp].prev_db = db;
                        stack[sp].prev_rb = rb;
                        ops[ops_len++] = OP_RNA;
                        cost = new_cost;
                        rb++;
                        sp++;
                    }
                }
            }
            continue;
        }

        if (trans == 2) {
            stack[top].trans = 3;
            if (dna_cost != INF_COST && db < MAX_DNA_BULGES && ctj > 0
                && cpi < PATTERN_LEN) {
                uint8_t prev_n = (cpi > 0)
                    ? (get_pattern(pattern_bit4, p, cpi - 1) == 0x0F) : 0;
                uint8_t next_n = (get_pattern(pattern_bit4, p, cpi) == 0x0F);
                if (!(prev_n && next_n)) {
                    uint8_t new_cost = cost + 1;
                    uint8_t rem = dp[cpi * (TEXT_WINDOW + 1) + (ctj - 1)];
                    if ((uint32_t)new_cost + (uint32_t)rem <= (uint32_t)MAX_EDITS
                        && ops_len < MAX_OPS) {
                        stack[sp].pi = cpi;
                        stack[sp].tj = ctj - 1;
                        stack[sp].trans = 0;
                        stack[sp].prev_ops_len = (uint8_t)ops_len;
                        stack[sp].prev_cost = cost;
                        stack[sp].prev_mm = mm;
                        stack[sp].prev_db = db;
                        stack[sp].prev_rb = rb;
                        ops[ops_len++] = OP_DNA;
                        cost = new_cost;
                        db++;
                        sp++;
                    }
                }
            }
            continue;
        }

        // trans == 3: all transitions tried; pop and restore parent state.
        ops_len = stack[top].prev_ops_len;
        cost = stack[top].prev_cost;
        mm = stack[top].prev_mm;
        db = stack[top].prev_db;
        rb = stack[top].prev_rb;
        sp--;
    }
}
