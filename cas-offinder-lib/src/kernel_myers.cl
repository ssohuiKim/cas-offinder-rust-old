// Myers bit-parallel sweep + semi-global DP + single-optimal traceback,
// all inside one kernel. Emits a compact `s_match` per accepted alignment
// (end_pos + pattern_idx + packed ops + counts). Host just unpacks ops bits
// and pulls chars from pattern/genome to rebuild the aligned strings — no
// host-side DP.
//
// Compile-time defines required:
//   PATTERN_LEN      - pattern length including PAM (e.g. 27)
//   PAM_LEN          - PAM region length (e.g. 3 for NGG)
//   MAX_EDITS        - max_mismatches + max_dna_bulges + max_rna_bulges
//   MAX_MISMATCHES   - per-type cap for substitutions
//   MAX_DNA_BULGES   - per-type cap for DNA bulges
//   MAX_RNA_BULGES   - per-type cap for RNA bulges
//   TEXT_WINDOW      - genome chars per work item (PATTERN_LEN + MAX_DNA_BULGES)
//   OUT_BUF_SIZE     - capacity of the output match buffer
//
// Op encoding (2 bits each, LSB of ops_packed = ops[0] = LAST op in alignment):
//   0 = Match, 1 = Substitution, 2 = DnaBulge (gap in pattern), 3 = RnaBulge

#define OP_MATCH 0
#define OP_SUB   1
#define OP_DNA   2
#define OP_RNA   3
#define INF_COST 100
#define PATTERN_BYTES (((PATTERN_LEN) + 1) / 2)
#define MAX_OPS (PATTERN_LEN + MAX_DNA_BULGES)
#define PEQ_STRIDE 5   // must match build_peq_array in search.rs

// cas-offinder match/mismatch rule (bit4):
//   tb4 == 0    → padding, mismatch
//   pb4 == 0xF  → pattern N wildcard, match
//   tb4 == 0xF  → genome N, matches only pattern N (so pb4 != 0xF → mismatch)
//   otherwise   → (pb4 & tb4) != 0
static inline uchar match_cost_bit4(uchar pb4, uchar tb4) {
    if (tb4 == 0)    return 1;
    if (pb4 == 0x0F) return 0;
    if (tb4 == 0x0F) return 1;
    return ((pb4 & tb4) != 0) ? 0 : 1;
}

// PAM-filter rule: filter N accepts any genome except padding; filter
// specific requires a specific genome base (rejects both padding and
// genome N).
static inline uchar pam_cell_ok(uchar g, uchar f) {
    if (f == 0x0F) return (g != 0) ? 1 : 0;
    if (g == 0 || g == 0x0F) return 0;
    return ((g & f) != 0) ? 1 : 0;
}

struct s_match {
    uint  chunk_idx;         // genome position of leftmost aligned text char
    uint  pattern_idx;
    ulong ops_packed;        // 2 bits per op, walk-back order
    uint  mismatches;
    ushort dna_bulge_size;
    ushort rna_bulge_size;
    uint  ops_count;
    uint  _pad;
};

static inline uchar get_bit4(__global const uchar* chunk, uint i) {
    uchar byte = chunk[i / 2];
    return (i % 2 == 0) ? (byte & 0x0F) : ((byte >> 4) & 0x0F);
}
static inline uchar get_pattern(__global const uchar* patterns, uint p, uint i) {
    uint base = p * PATTERN_BYTES;
    uchar byte = patterns[base + i / 2];
    return (i % 2 == 0) ? (byte & 0x0F) : ((byte >> 4) & 0x0F);
}

__kernel void find_matches_myers(
    __global const uchar* genome_bit4,
    __global const ulong* peq_tables,
    __global const uchar* pattern_bit4,      // packed bit4 patterns, PATTERN_BYTES each
    __global const uchar* pam_offsets,       // per-pattern
    __global const uchar* pam_filters_bit4,  // per-pattern, PAM_LEN entries
    uint n_patterns,
    uint n_fwd_patterns,
    uint active_start_nucl,                  // skip positions < this (head overlap)
    uint total_nucl,
    __global struct s_match* out_matches,
    __global uint* out_count
) {
    uint j = get_global_id(0);   // alignment end position (inclusive)
    uint p = get_global_id(1);   // pattern index

    if (p >= n_patterns) return;
    if (j >= total_nucl) return;
    if (j < active_start_nucl) return;       // skip head-overlap positions
    if (j + 1 < PATTERN_LEN) return;

    // ---- PAM pre-check: try each possible DNA bulge shift (for PAM-first
    // patterns PAM position in genome shifts with bulge count; for PAM-last
    // it's anchored to j). If any shift has the PAM bases present, accept.
    uint pam_off = (uint)pam_offsets[p];
    uint pam_shift_range = (pam_off == 0) ? (MAX_DNA_BULGES + 1u) : 1u;
    bool pam_ok = false;
    for (uint b = 0; b < pam_shift_range; b++) {
        if (j + 1 < PATTERN_LEN + b) continue;
        uint align_start_try = j + 1 - PATTERN_LEN - b;
        bool this_ok = true;
        for (uint k = 0; k < PAM_LEN; k++) {
            uint gpos = align_start_try + pam_off + k;
            if (gpos >= total_nucl) { this_ok = false; break; }
            uchar g = get_bit4(genome_bit4, gpos);
            uchar f = pam_filters_bit4[p * PAM_LEN + k];
            if (!pam_cell_ok(g, f)) { this_ok = false; break; }
        }
        if (this_ok) { pam_ok = true; break; }
    }
    if (!pam_ok) return;

    // ---- Myers sweep over the text window to bound final edit distance ----
    ulong peq_a = peq_tables[p * PEQ_STRIDE + 0];
    ulong peq_c = peq_tables[p * PEQ_STRIDE + 1];
    ulong peq_g = peq_tables[p * PEQ_STRIDE + 2];
    ulong peq_t = peq_tables[p * PEQ_STRIDE + 3];
    ulong peq_n = peq_tables[p * PEQ_STRIDE + 4];

    ulong mask = (PATTERN_LEN < 64) ? (((ulong)1 << PATTERN_LEN) - 1) : ~(ulong)0;
    ulong last_bit = (ulong)1 << (PATTERN_LEN - 1);
    ulong vp = mask;
    ulong vn = 0;
    int score = PATTERN_LEN;

    uint text_start = (j + 1 > (uint)TEXT_WINDOW) ? (j + 1 - (uint)TEXT_WINDOW) : 0;
    uint text_len = j + 1 - text_start;  // <= TEXT_WINDOW
    for (uint t = text_start; t <= j; t++) {
        uchar b4 = get_bit4(genome_bit4, t);
        ulong eq;
        if (b4 == 0) {
            eq = 0;
        } else if (b4 == 0x0F) {
            // genome N: match only pattern N (cas-offinder C++ semantics)
            eq = peq_n;
        } else {
            eq = 0;
            if (b4 & 0x4) eq |= peq_a;
            if (b4 & 0x2) eq |= peq_c;
            if (b4 & 0x8) eq |= peq_g;
            if (b4 & 0x1) eq |= peq_t;
        }

        ulong x = eq | vn;
        ulong d0 = (((x & vp) + vp) ^ vp) | x;
        ulong hn = vp & d0;
        ulong hp = vn | ~(vp | d0);
        ulong x_shift = hp << 1;
        vn = x_shift & d0;
        vp = (hn << 1) | ~(x_shift | d0);
        vp &= mask;
        vn &= mask;
        if (hp & last_bit) score++;
        if (hn & last_bit) score--;
    }
    if (score < 0 || score > (int)MAX_EDITS) return;

    // ---- Full DP in private memory. cells are uchar (max edits well under 255). ----
    uchar dna_cost = (MAX_DNA_BULGES == 0) ? INF_COST : 1;
    uchar rna_cost = (MAX_RNA_BULGES == 0) ? INF_COST : 1;

    uchar dp[(PATTERN_LEN + 1) * (TEXT_WINDOW + 1)];

    // dp[i][0] = i * rna_cost (saturated)
    for (uint i = 0; i <= PATTERN_LEN; i++) {
        uint v = i * (uint)rna_cost;
        dp[i * (TEXT_WINDOW + 1)] = (v > INF_COST) ? (uchar)INF_COST : (uchar)v;
    }
    // dp[0][t] = 0 for semi-global (text prefix free)
    for (uint t = 0; t <= TEXT_WINDOW; t++) {
        dp[t] = 0;
    }
    // Only fill up through `text_len` so trailing cells at end are defined.
    // Text bytes with bit4 == 0 mark chromosome-boundary padding (FASTA reader
    // zero-fills the tail of partial chunks). Force such cells to INF_COST so
    // no alignment path can cross into padding — otherwise candidates that
    // straddle the gap between concatenated chromosomes leak into the output.
    for (uint pi = 1; pi <= PATTERN_LEN; pi++) {
        uchar pb4 = get_pattern(pattern_bit4, p, pi - 1);
        for (uint tj = 1; tj <= TEXT_WINDOW; tj++) {
            if (tj > text_len) {
                dp[pi * (TEXT_WINDOW + 1) + tj] = INF_COST;
                continue;
            }
            uchar tb4 = get_bit4(genome_bit4, text_start + tj - 1);
            if (tb4 == 0) {
                dp[pi * (TEXT_WINDOW + 1) + tj] = INF_COST;
                continue;
            }
            uchar mc = match_cost_bit4(pb4, tb4);
            uint diag = (uint)dp[(pi - 1) * (TEXT_WINDOW + 1) + (tj - 1)] + mc;
            uint up   = (uint)dp[(pi - 1) * (TEXT_WINDOW + 1) + tj] + (uint)rna_cost;
            uint left = (uint)dp[pi       * (TEXT_WINDOW + 1) + (tj - 1)] + (uint)dna_cost;
            uint v = min(diag, min(up, left));
            if (v > INF_COST) v = INF_COST;
            dp[pi * (TEXT_WINDOW + 1) + tj] = (uchar)v;
        }
    }

    uint last_cell = PATTERN_LEN * (TEXT_WINDOW + 1) + text_len;
    uchar final_cost = dp[last_cell];
    if (final_cost > (uchar)MAX_EDITS) return;

    // ---- Multi-path iterative DFS traceback.
    //
    // Enumerates every alignment with (mm <= MAX_MISMATCHES, db <= MAX_DNA_BULGES,
    // rb <= MAX_RNA_BULGES, cost <= MAX_EDITS) by walking all valid backward
    // transitions from (PATTERN_LEN, text_len) to (0, any). Pruning uses
    // dp[i][j] as lower bound on remaining cost from (i, j) to (0, *) so the
    // search visits only cells that can lead to at least one valid alignment.
    //
    // Frame layout (8 B) holds position + next transition to try + parent-state
    // undo info (restore on pop so ops[] and per-type counters mirror recursion).
    struct frame {
        uchar pi;
        uchar tj;
        uchar trans;           // 0=diag, 1=RNA, 2=DNA, 3=done (pop)
        uchar prev_ops_len;
        uchar prev_cost;
        uchar prev_mm;
        uchar prev_db;
        uchar prev_rb;
    };
    // Max DFS depth = pattern-decrementing transitions + DNA bulges
    //   = PATTERN_LEN + MAX_DNA_BULGES (RNA bulges count toward pattern-dec).
    struct frame stack[PATTERN_LEN + MAX_DNA_BULGES + 1];
    uchar ops[MAX_OPS];
    uint sp = 0;
    uint ops_len = 0;
    uchar cost = 0, mm = 0, db = 0, rb = 0;

    stack[0].pi = PATTERN_LEN;
    stack[0].tj = (uchar)text_len;
    stack[0].trans = 0;
    stack[0].prev_ops_len = 0;
    stack[0].prev_cost = 0;
    stack[0].prev_mm = 0;
    stack[0].prev_db = 0;
    stack[0].prev_rb = 0;
    sp = 1;

    while (sp > 0) {
        uint top = sp - 1;
        uchar cpi = stack[top].pi;
        uchar ctj = stack[top].tj;

        if (cpi == 0) {
            // ---- Leaf: emit this alignment. ----
            // genome_span = non-gap text chars = PATTERN_LEN - rb + db
            uint genome_span = (uint)PATTERN_LEN + (uint)db - (uint)rb;
            uint match_start = j + 1 - genome_span;

            // PAM verification walking ops forward (reverse of push order).
            uint pi_fwd = 0;
            uint g_off = 0;
            bool pam_ok_full = true;
            for (int oi = (int)ops_len - 1; oi >= 0; oi--) {
                uchar op = ops[oi];
                if (op == OP_MATCH || op == OP_SUB) {
                    uchar pb4 = get_pattern(pattern_bit4, p, pi_fwd);
                    if (pb4 == 0x0F) {
                        uchar gb4 = get_bit4(genome_bit4, match_start + g_off);
                        uint k = pi_fwd - pam_off;
                        uchar f = pam_filters_bit4[p * PAM_LEN + k];
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
                ulong ops_packed = 0;
                for (uint k = 0; k < ops_len; k++) {
                    ops_packed |= ((ulong)ops[k]) << (k * 2);
                }
                uint idx = atomic_inc(out_count);
                if (idx < OUT_BUF_SIZE) {
                    out_matches[idx].chunk_idx = match_start;
                    out_matches[idx].pattern_idx = p;
                    out_matches[idx].ops_packed = ops_packed;
                    out_matches[idx].mismatches = mm;
                    out_matches[idx].dna_bulge_size = (ushort)db;
                    out_matches[idx].rna_bulge_size = (ushort)rb;
                    out_matches[idx].ops_count = ops_len;
                    out_matches[idx]._pad = 0;
                }
            }

            // Pop leaf and restore parent state.
            ops_len = stack[top].prev_ops_len;
            cost = stack[top].prev_cost;
            mm = stack[top].prev_mm;
            db = stack[top].prev_db;
            rb = stack[top].prev_rb;
            sp--;
            continue;
        }

        uchar trans = stack[top].trans;

        // ---- Try diagonal (match / substitution). ----
        if (trans == 0) {
            stack[top].trans = 1;
            if (ctj > 0) {
                uchar pb4 = get_pattern(pattern_bit4, p, cpi - 1);
                uchar tb4 = get_bit4(genome_bit4, text_start + ctj - 1);
                uchar mc = match_cost_bit4(pb4, tb4);
                if (mc == 0 || pb4 != 0x0F) {
                    uchar new_cost = cost + mc;
                    uchar new_mm = mm + mc;
                    uchar rem = dp[(cpi - 1) * (TEXT_WINDOW + 1) + (ctj - 1)];
                    if ((uint)new_cost + (uint)rem <= (uint)MAX_EDITS
                        && new_mm <= MAX_MISMATCHES
                        && ops_len < MAX_OPS) {
                        // Save parent state in the new child frame; then apply transition.
                        stack[sp].pi = cpi - 1;
                        stack[sp].tj = ctj - 1;
                        stack[sp].trans = 0;
                        stack[sp].prev_ops_len = (uchar)ops_len;
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

        // ---- Try RNA bulge (gap in text). ----
        if (trans == 1) {
            stack[top].trans = 2;
            if (rna_cost != INF_COST && rb < MAX_RNA_BULGES) {
                uchar pb4 = get_pattern(pattern_bit4, p, cpi - 1);
                if (pb4 != 0x0F) {
                    uchar new_cost = cost + 1;
                    uchar rem = dp[(cpi - 1) * (TEXT_WINDOW + 1) + ctj];
                    if ((uint)new_cost + (uint)rem <= (uint)MAX_EDITS
                        && ops_len < MAX_OPS) {
                        stack[sp].pi = cpi - 1;
                        stack[sp].tj = ctj;
                        stack[sp].trans = 0;
                        stack[sp].prev_ops_len = (uchar)ops_len;
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

        // ---- Try DNA bulge (gap in pattern). ----
        //
        // Matches cas-offinder-bulge's rule: allow a bulge between pattern
        // positions cpi-1 and cpi iff:
        //   * not strictly inside the PAM (both neighbours == N), and
        //   * not past the PAM-side end of the pattern (cpi == PATTERN_LEN).
        // This mirrors traceback.rs:335 for CPU/GPU parity.
        if (trans == 2) {
            stack[top].trans = 3;
            if (dna_cost != INF_COST && db < MAX_DNA_BULGES && ctj > 0
                && cpi < PATTERN_LEN) {
                uchar prev_n = (cpi > 0) ? (get_pattern(pattern_bit4, p, cpi - 1) == 0x0F) : 0;
                uchar next_n = (get_pattern(pattern_bit4, p, cpi) == 0x0F);
                if (!(prev_n && next_n)) {
                    uchar new_cost = cost + 1;
                    uchar rem = dp[cpi * (TEXT_WINDOW + 1) + (ctj - 1)];
                    if ((uint)new_cost + (uint)rem <= (uint)MAX_EDITS
                        && ops_len < MAX_OPS) {
                        stack[sp].pi = cpi;
                        stack[sp].tj = ctj - 1;
                        stack[sp].trans = 0;
                        stack[sp].prev_ops_len = (uchar)ops_len;
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
