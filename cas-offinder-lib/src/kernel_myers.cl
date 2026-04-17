// Myers bit-parallel edit distance kernel with PAM pre-filter.
//
// Each work item first checks whether the PAM (e.g. NGG) is present at the
// expected genome position. If not, it returns immediately (~2 ops). Only
// positions that pass the PAM check run the full Myers sweep (~pattern_len × 7
// bit-ops). Since PAM occurs at roughly 1/16 of positions, this eliminates
// ~94% of work.
//
// Compile-time defines required:
//   PATTERN_LEN   - full pattern length including PAM mask (e.g. 27)
//   PAM_LEN       - length of PAM region (e.g. 3 for NGG)
//   MAX_EDITS     - maximum edit distance to report
//   TEXT_WINDOW   - genome chars consumed per work item (PATTERN_LEN + max_dna_bulges)
//   OUT_BUF_SIZE  - capacity of the output match buffer

struct s_match {
    uint chunk_idx;
    uint pattern_idx;
    uint mismatches;
    ushort dna_bulge_size;
    ushort rna_bulge_size;
};

static inline uchar get_bit4(__global const uchar* chunk, uint i) {
    uchar byte = chunk[i / 2];
    return (i % 2 == 0) ? (byte & 0x0F) : ((byte >> 4) & 0x0F);
}

__kernel void find_matches_myers(
    __global const uchar* genome_bit4,
    __global const ulong* peq_tables,       // 4 u64s per pattern
    __global const uchar* pam_offsets,       // per-pattern: offset from align_start to PAM start
    __global const uchar* pam_filters_bit4,  // per-pattern: PAM_LEN bytes of bit4 filter
    uint n_patterns,
    uint n_fwd_patterns,
    uint total_nucl,
    __global struct s_match* out_matches,
    __global uint* out_count
) {
    uint j = get_global_id(0);   // alignment end position (inclusive)
    uint p = get_global_id(1);   // pattern index

    if (p >= n_patterns) return;
    if (j >= total_nucl) return;
    if (j + 1 < PATTERN_LEN) return;

    // ---- PAM pre-check (cheap: ~PAM_LEN × 2 ops) ----
    // DNA bulges shift the alignment start leftward; for PAM-first patterns
    // (pam_off == 0) the PAM position depends on the candidate's bulge count,
    // so we try each possible shift in [0, MAX_DNA_BULGES]. For PAM-last
    // patterns the PAM is anchored to j and unaffected by bulges.
    uint pam_off = (uint)pam_offsets[p];
    uint pam_shift_range = (pam_off == 0) ? (MAX_DNA_BULGES + 1u) : 1u;
    bool pam_ok = false;
    for (uint b = 0; b < pam_shift_range; b++) {
        if (j + 1 < PATTERN_LEN + b) continue;
        uint align_start = j + 1 - PATTERN_LEN - b;
        bool this_ok = true;
        for (uint k = 0; k < PAM_LEN; k++) {
            uint gpos = align_start + pam_off + k;
            if (gpos >= total_nucl) { this_ok = false; break; }
            uchar g = get_bit4(genome_bit4, gpos);
            uchar f = pam_filters_bit4[p * PAM_LEN + k];
            if ((g & f) == 0) { this_ok = false; break; }
        }
        if (this_ok) { pam_ok = true; break; }
    }
    if (!pam_ok) return;

    // ---- Myers bit-parallel (only reached ~1/16 of the time) ----
    ulong peq_a = peq_tables[p * 4 + 0];
    ulong peq_c = peq_tables[p * 4 + 1];
    ulong peq_g = peq_tables[p * 4 + 2];
    ulong peq_t = peq_tables[p * 4 + 3];

    ulong mask = (PATTERN_LEN < 64) ? (((ulong)1 << PATTERN_LEN) - 1) : ~(ulong)0;
    ulong last_bit = (ulong)1 << (PATTERN_LEN - 1);
    ulong vp = mask;
    ulong vn = 0;
    int score = PATTERN_LEN;

    uint window_len = TEXT_WINDOW;
    uint text_start = (j + 1 > window_len) ? (j + 1 - window_len) : 0;

    for (uint t = text_start; t <= j; t++) {
        uchar b4 = get_bit4(genome_bit4, t);
        ulong eq = 0;
        if (b4 & 0x4) eq |= peq_a;
        if (b4 & 0x2) eq |= peq_c;
        if (b4 & 0x8) eq |= peq_g;
        if (b4 & 0x1) eq |= peq_t;

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

    if (score >= 0 && score <= (int)MAX_EDITS) {
        uint idx = atomic_inc(out_count);
        if (idx < OUT_BUF_SIZE) {
            out_matches[idx].chunk_idx = j;
            out_matches[idx].pattern_idx = p;
            out_matches[idx].mismatches = (uint)score;
            out_matches[idx].dna_bulge_size = 0;
            out_matches[idx].rna_bulge_size = 0;
        }
    }
}
