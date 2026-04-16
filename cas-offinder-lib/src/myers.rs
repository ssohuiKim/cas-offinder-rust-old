// cas-offinder-lib/src/myers.rs
//! Myers bit-parallel edit distance algorithm (1999).
//! Supports patterns up to 64 characters using a single u64 word.

/// Peq table: for each base (A, C, G, T), a bit-vector indicating
/// which pattern positions match that base. Pattern length <= 64.
#[derive(Debug, Clone, Copy)]
pub struct PeqTable {
    pub peq: [u64; 4], // indexed by nucl_idx: A=0, C=1, G=2, T=3
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
    for (i, &c) in pattern.iter().enumerate() {
        let idx = nucl_idx(c);
        if idx < 4 {
            peq[idx as usize] |= 1u64 << i;
        } else {
            for p in peq.iter_mut() {
                *p |= 1u64 << i;
            }
        }
    }
    PeqTable { peq, pattern_len: pattern.len() }
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
            peq.peq[0] | peq.peq[1] | peq.peq[2] | peq.peq[3]
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nucl_idx() {
        assert_eq!(nucl_idx(b'A'), 0);
        assert_eq!(nucl_idx(b'C'), 1);
        assert_eq!(nucl_idx(b'G'), 2);
        assert_eq!(nucl_idx(b'T'), 3);
        assert_eq!(nucl_idx(b'a'), 0);
        assert_eq!(nucl_idx(b'N'), 4);
    }

    #[test]
    fn test_myers_exact_match() {
        let pattern = b"ACGT";
        let peq = build_peq(pattern);
        assert_eq!(myers_edit_distance(&peq, b"ACGT"), 0);
    }

    #[test]
    fn test_myers_exact_match_at_end() {
        let pattern = b"ACGT";
        let peq = build_peq(pattern);
        assert_eq!(myers_edit_distance(&peq, b"TTACGT"), 0);
    }

    #[test]
    fn test_myers_exact_match_with_n_text() {
        let pattern = b"ACGT";
        let peq = build_peq(pattern);
        assert_eq!(myers_edit_distance(&peq, b"ACGN"), 0);
    }

    #[test]
    fn test_myers_one_substitution() {
        let pattern = b"ACGT";
        let peq = build_peq(pattern);
        assert_eq!(myers_edit_distance(&peq, b"AAGT"), 1);
        assert_eq!(myers_edit_distance(&peq, b"ACTT"), 1);
    }

    #[test]
    fn test_myers_multiple_substitutions() {
        let pattern = b"ACGT";
        let peq = build_peq(pattern);
        assert_eq!(myers_edit_distance(&peq, b"AAGA"), 2);
        assert_eq!(myers_edit_distance(&peq, b"ATTC"), 3);
    }

    #[test]
    fn test_myers_long_pattern_substitutions() {
        let pattern = b"ACGTACGTACGTACGTACGT";
        let peq = build_peq(pattern);
        assert_eq!(myers_edit_distance(&peq, b"ACGTACGTACGTACGTACGT"), 0);
        assert_eq!(myers_edit_distance(&peq, b"ACGTAAGTACGTACGTACGT"), 1);
        assert_eq!(myers_edit_distance(&peq, b"ACGTACGTACGTACGTACGA"), 1);
    }

    #[test]
    fn test_myers_one_dna_bulge() {
        let pattern = b"ACGTACGT";
        let peq = build_peq(pattern);
        assert_eq!(myers_edit_distance(&peq, b"ACGTTACGT"), 1);
        assert_eq!(myers_edit_distance(&peq, b"TACGTACGT"), 0);
    }

    #[test]
    fn test_myers_two_dna_bulges() {
        let pattern = b"ACGTACGT";
        let peq = build_peq(pattern);
        assert_eq!(myers_edit_distance(&peq, b"ACGNTAACGT"), 2);
    }

    #[test]
    fn test_myers_one_rna_bulge() {
        let pattern = b"ACGTACGT";
        let peq = build_peq(pattern);
        assert_eq!(myers_edit_distance(&peq, b"ACGACGT"), 1);
    }

    #[test]
    fn test_myers_two_rna_bulges() {
        let pattern = b"ACGTACGT";
        let peq = build_peq(pattern);
        assert_eq!(myers_edit_distance(&peq, b"ACGACG"), 2);
    }

    #[test]
    fn test_myers_mixed_edits() {
        let pattern = b"ACGTACGT";
        let peq = build_peq(pattern);
        // pattern:  A C G T A C G T
        // text:     A C G T X A A G T  (X insertion, C→A mismatch)
        assert_eq!(myers_edit_distance(&peq, b"ACGTXAAGT"), 2);
    }

    #[test]
    fn test_myers_worst_case() {
        let pattern = b"ACGTACGT";
        let peq = build_peq(pattern);
        let result = myers_edit_distance(&peq, b"TTTTTTTT");
        assert!(result >= 4 && result <= 8, "result was {}", result);
    }

    // Reference implementation: standard edit distance DP (semi-global).
    fn reference_edit_distance(pattern: &[u8], text: &[u8]) -> u32 {
        let m = pattern.len();
        let n = text.len();
        let mut dp = vec![vec![0u32; n + 1]; m + 1];
        for i in 0..=m { dp[i][0] = i as u32; }
        for i in 1..=m {
            for j in 1..=n {
                let match_cost = if cmp_char_loose(pattern[i-1], text[j-1]) { 0 } else { 1 };
                dp[i][j] = std::cmp::min(
                    dp[i-1][j-1] + match_cost,
                    std::cmp::min(dp[i-1][j] + 1, dp[i][j-1] + 1),
                );
            }
        }
        dp[m][n]
    }

    fn cmp_char_loose(pc: u8, tc: u8) -> bool {
        let pu = pc.to_ascii_uppercase();
        let tu = tc.to_ascii_uppercase();
        if pu == b'N' || tu == b'N' { return true; }
        pu == tu
    }

    #[test]
    fn test_myers_matches_reference_random() {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let bases = [b'A', b'C', b'G', b'T'];
        for seed in 0..50u64 {
            let mut h = DefaultHasher::new();
            seed.hash(&mut h);
            let mut state = h.finish();
            let m = 8 + (seed % 17) as usize;
            let n = m + (seed % 10) as usize;
            let mut next = || {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                (state >> 32) as u32
            };
            let pattern: Vec<u8> = (0..m).map(|_| bases[(next() as usize) % 4]).collect();
            let text: Vec<u8> = (0..n).map(|_| bases[(next() as usize) % 4]).collect();
            let peq = build_peq(&pattern);
            let myers_result = myers_edit_distance(&peq, &text);
            let ref_result = reference_edit_distance(&pattern, &text);
            assert_eq!(myers_result, ref_result,
                "pattern={:?} text={:?}",
                std::str::from_utf8(&pattern).unwrap(),
                std::str::from_utf8(&text).unwrap());
        }
    }
}
