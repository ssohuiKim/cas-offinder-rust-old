// cas-offinder-lib/src/myers.rs
//! Myers bit-parallel edit distance algorithm (1999).
//! Supports patterns up to 64 characters using a single u64 word.

/// Peq table: for each base (A, C, G, T), a bit-vector indicating
/// which pattern positions match that base. Pattern length <= 64.
/// `peq_n` marks positions where the pattern holds 'N' (wildcard). When a
/// genome position is itself 'N' (bit4 == 0xF) the Myers sweep uses only
/// `peq_n`, so genome 'N' matches pattern 'N' and mismatches pattern
/// specific bases — matching cas-offinder's C++ convention.
#[derive(Debug, Clone, Copy)]
pub struct PeqTable {
    pub peq: [u64; 4], // indexed by nucl_idx: A=0, C=1, G=2, T=3
    pub peq_n: u64,
    pub pattern_len: usize,
}

/// Convert nucleotide ASCII char to index (A=0, C=1, G=2, T=3).
/// Returns 4 for N/ambiguous (matches all).
fn nucl_idx(c: u8) -> u8 {
    match c.to_ascii_uppercase() {
        b'A' => 0,
        b'C' => 1,
        b'G' => 2,
        b'T' => 3,
        _ => 4, // N or ambiguous
    }
}

/// Build a Peq table from an ASCII pattern.
/// Characters that aren't exact A/C/G/T match everything (like N).
pub fn build_peq(pattern: &[u8]) -> PeqTable {
    assert!(pattern.len() <= 64, "Myers requires pattern length <= 64");
    let mut peq = [0u64; 4];
    let mut peq_n = 0u64;
    for (i, &c) in pattern.iter().enumerate() {
        let idx = nucl_idx(c);
        if idx < 4 {
            peq[idx as usize] |= 1u64 << i;
        } else {
            // Wildcard (N/IUPAC): match all 4 concrete bases AND count as
            // match when the genome itself is an 'N'.
            for p in peq.iter_mut() {
                *p |= 1u64 << i;
            }
            peq_n |= 1u64 << i;
        }
    }
    PeqTable { peq, peq_n, pattern_len: pattern.len() }
}

/// Compute minimum edit distance: pattern must fully match some suffix of
/// text (semi-global: free end gaps on text start only). Returns distance.
pub fn myers_edit_distance(peq: &PeqTable, text: &[u8]) -> u32 {
    let m = peq.pattern_len;
    if m == 0 { return 0; }

    let mask = if m < 64 { (1u64 << m) - 1 } else { !0u64 };
    let last_bit = 1u64 << (m - 1);
    let mut vp: u64 = mask;
    let mut vn: u64 = 0;
    let mut score: i32 = m as i32;

    for &c in text {
        let idx = nucl_idx(c);
        let eq = if idx < 4 {
            peq.peq[idx as usize]
        } else {
            // Genome 'N': per cas-offinder C++, matches only pattern 'N'.
            peq.peq_n
        };
        let x = eq | vn;
        let d0 = (((x & vp).wrapping_add(vp)) ^ vp) | x;
        let hn = vp & d0;
        let hp = vn | !(vp | d0);
        // Semi-global: no `| 1` (free leading gaps in text, i.e. dp[0][j] = 0)
        let x_shift = hp << 1;
        vn = x_shift & d0;
        vp = (hn << 1) | !(x_shift | d0);
        // Mask to pattern length to avoid dirty high bits
        vp &= mask;
        vn &= mask;
        if hp & last_bit != 0 { score += 1; }
        if hn & last_bit != 0 { score -= 1; }
    }
    score as u32
}

