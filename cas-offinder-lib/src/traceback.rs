// cas-offinder-lib/src/traceback.rs
//! Alignment reconstruction (traceback) using standard DP.
//! Only called on Myers-identified candidates, so total cost is small.

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EditOp {
    Match,
    Substitution,
    DnaBulge, // insertion in text (genome)
    RnaBulge, // deletion in text (pattern has extra char)
}

#[derive(Debug, Clone)]
pub struct Alignment {
    pub ops: Vec<EditOp>,
    pub pattern_aligned: Vec<u8>, // with '-' for gaps
    pub text_aligned: Vec<u8>,    // with '-' for gaps
    pub text_start: usize,        // text position where alignment starts
    pub mismatches: u32,
    pub dna_bulges: u32,          // number of bulge positions (size, not count)
    pub rna_bulges: u32,
}

fn cmp_loose(pc: u8, tc: u8) -> bool {
    let pu = pc.to_ascii_uppercase();
    let tu = tc.to_ascii_uppercase();
    if pu == b'N' || tu == b'N' { return true; }
    pu == tu
}

/// Reconstruct the best alignment of `pattern` ending at text position `text.len()`.
/// Semi-global: pattern must be fully matched; text start is free.
/// Returns None if no alignment found with edit distance <= max_edits.
pub fn traceback(pattern: &[u8], text: &[u8], max_edits: u32) -> Option<Alignment> {
    let m = pattern.len();
    let n = text.len();

    let mut dp = vec![vec![0u32; n + 1]; m + 1];
    for i in 0..=m { dp[i][0] = i as u32; }
    // dp[0][j] = 0 (semi-global)

    for i in 1..=m {
        for j in 1..=n {
            let match_cost = if cmp_loose(pattern[i-1], text[j-1]) { 0 } else { 1 };
            dp[i][j] = std::cmp::min(
                dp[i-1][j-1] + match_cost,
                std::cmp::min(dp[i-1][j] + 1, dp[i][j-1] + 1),
            );
        }
    }

    let final_dist = dp[m][n];
    if final_dist > max_edits { return None; }

    // Traceback from dp[m][n] backward
    let mut ops = Vec::new();
    let mut pa = Vec::new();
    let mut ta = Vec::new();
    let mut i = m;
    let mut j = n;
    let mut mismatches = 0u32;
    let mut dna_bulges = 0u32;
    let mut rna_bulges = 0u32;

    while i > 0 || j > 0 {
        if i > 0 && j > 0 {
            let match_cost = if cmp_loose(pattern[i-1], text[j-1]) { 0 } else { 1 };
            if dp[i][j] == dp[i-1][j-1] + match_cost {
                // Match or substitution
                if match_cost == 0 {
                    ops.push(EditOp::Match);
                } else {
                    ops.push(EditOp::Substitution);
                    mismatches += 1;
                }
                pa.push(pattern[i-1]);
                ta.push(text[j-1]);
                i -= 1; j -= 1;
                continue;
            }
            if dp[i][j] == dp[i-1][j] + 1 {
                // Deletion: pattern has extra character (RNA bulge)
                ops.push(EditOp::RnaBulge);
                rna_bulges += 1;
                pa.push(pattern[i-1]);
                ta.push(b'-');
                i -= 1;
                continue;
            }
            if dp[i][j] == dp[i][j-1] + 1 {
                // Insertion: text has extra character (DNA bulge)
                ops.push(EditOp::DnaBulge);
                dna_bulges += 1;
                pa.push(b'-');
                ta.push(text[j-1]);
                j -= 1;
                continue;
            }
        }
        if i > 0 && (j == 0 || dp[i][j] == dp[i-1][j] + 1) {
            ops.push(EditOp::RnaBulge);
            rna_bulges += 1;
            pa.push(pattern[i-1]);
            ta.push(b'-');
            i -= 1;
            continue;
        }
        if j > 0 {
            // Semi-global free start: unmatched text prefix
            // Don't record this as an op; just advance j
            j -= 1;
            continue;
        }
    }

    ops.reverse();
    pa.reverse();
    ta.reverse();

    Some(Alignment {
        ops,
        pattern_aligned: pa,
        text_aligned: ta,
        text_start: j,
        mismatches,
        dna_bulges,
        rna_bulges,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_traceback_exact() {
        let a = traceback(b"ACGT", b"ACGT", 2).unwrap();
        assert_eq!(a.mismatches, 0);
        assert_eq!(a.dna_bulges, 0);
        assert_eq!(a.rna_bulges, 0);
        assert_eq!(a.pattern_aligned, b"ACGT");
        assert_eq!(a.text_aligned, b"ACGT");
    }

    #[test]
    fn test_traceback_substitution() {
        let a = traceback(b"ACGT", b"AAGT", 2).unwrap();
        assert_eq!(a.mismatches, 1);
        assert_eq!(a.dna_bulges, 0);
        assert_eq!(a.rna_bulges, 0);
    }

    #[test]
    fn test_traceback_dna_bulge() {
        // text has extra X: "ACGT" vs "ACXGT"
        let a = traceback(b"ACGT", b"ACXGT", 2).unwrap();
        assert_eq!(a.dna_bulges, 1);
        assert_eq!(a.mismatches, 0);
        assert_eq!(a.rna_bulges, 0);
        assert_eq!(a.pattern_aligned, b"AC-GT");
        assert_eq!(a.text_aligned, b"ACXGT");
    }

    #[test]
    fn test_traceback_rna_bulge() {
        // pattern has extra char: "ACGT" vs "ACT" (G deleted from text)
        let a = traceback(b"ACGT", b"ACT", 2).unwrap();
        assert_eq!(a.rna_bulges, 1);
        assert_eq!(a.mismatches, 0);
        assert_eq!(a.dna_bulges, 0);
        assert_eq!(a.pattern_aligned, b"ACGT");
        assert_eq!(a.text_aligned, b"AC-T");
    }

    #[test]
    fn test_traceback_exceeds_max() {
        let result = traceback(b"ACGT", b"TTTT", 1);
        assert!(result.is_none(), "Should not find alignment with 1 edit");
    }
}
