// Myers bit-parallel edit distance kernel.
// Each work item computes the edit distance between one pattern (indexed by
// get_global_id(1)) and the genome window ending at position get_global_id(0).
// Patterns up to 64 nucleotides fit in a single u64 Peq bit-vector.
//
// Compile-time defines required:
//   PATTERN_LEN   - length of pattern in nucleotides (<= 64)
//   MAX_EDITS     - maximum edit distance to report (runtime filter)
//   TEXT_WINDOW   - number of genome chars consumed per work item
//                   (usually PATTERN_LEN + max_dna_bulges)
//   OUT_BUF_SIZE  - capacity of the output match buffer
//
// Genome is passed as 4-bit packed nucleotides (two per byte), matching the
// CPU encoding. Ambiguous bases (N, bit-4 value 0xF) match every pattern
// position; zero bytes (padding beyond active region) match nothing.

struct s_match {
    uint chunk_idx;     // end position (exclusive) of alignment in chunk
    uint pattern_idx;   // pattern index
    uint mismatches;    // edit distance (may include bulges, CPU reclassifies)
    ushort dna_bulge_size;
    ushort rna_bulge_size;
};

static inline uchar get_bit4(__global const uchar* chunk, uint i) {
    uchar byte = chunk[i / 2];
    return (i % 2 == 0) ? (byte & 0x0F) : ((byte >> 4) & 0x0F);
}

__kernel void find_matches_myers(
    __global const uchar* genome_bit4,
    __global const ulong* peq_tables,    // 4 u64s per pattern
    uint n_patterns,
    uint total_nucl,
    __global struct s_match* out_matches,
    __global uint* out_count
) {
    uint j = get_global_id(0);   // alignment end position (exclusive) - 1
    uint p = get_global_id(1);   // pattern index

    if (p >= n_patterns) return;
    if (j >= total_nucl) return;
    if (j + 1 < PATTERN_LEN) return;

    // Load Peq for this pattern
    ulong peq_a = peq_tables[p * 4 + 0];
    ulong peq_c = peq_tables[p * 4 + 1];
    ulong peq_g = peq_tables[p * 4 + 2];
    ulong peq_t = peq_tables[p * 4 + 3];

    ulong mask = (PATTERN_LEN < 64) ? (((ulong)1 << PATTERN_LEN) - 1) : ~(ulong)0;
    ulong last_bit = (ulong)1 << (PATTERN_LEN - 1);
    ulong vp = mask;
    ulong vn = 0;
    int score = PATTERN_LEN;

    // Text window ending at (and including) position j.
    uint window_len = TEXT_WINDOW;
    uint text_start = (j + 1 > window_len) ? (j + 1 - window_len) : 0;

    for (uint t = text_start; t <= j; t++) {
        uchar b4 = get_bit4(genome_bit4, t);
        ulong eq = 0;
        // bit4 encoding: A=0x4, C=0x2, G=0x8, T=0x1. b4==0 means padding (no match).
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
