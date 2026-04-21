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

/// Bit4 match following cas-offinder C++ semantics:
///   * `t == 0`           (padding / outside valid genome) → never match
///   * `p == 0xF` (pat N) → always match (wildcard)
///   * `t == 0xF` (gen N) + pattern specific base → **mismatch** (genome N
///     only matches pattern N, not A/C/G/T)
///   * otherwise `(p & t) != 0`
///
/// Must stay consistent with the GPU kernel, the Myers eq computation, and
/// the PAM filter — CPU/GPU parity depends on it.
#[inline(always)]
fn cmp_loose_bit4(p: u8, t: u8) -> bool {
    if t == 0 { return false; }
    if p == 0xF { return true; }
    if t == 0xF { return false; }
    (p & t) != 0
}

/// Reconstruct the best alignment of `pattern` ending at text position `text.len()`.
/// Semi-global: pattern must be fully matched; text start is free.
/// Returns None if no alignment found with edit distance <= max_edits.
///
/// `max_dna_bulges` / `max_rna_bulges`: per-type bulge caps. When a cap is 0,
/// that bulge type is forbidden entirely in the DP (its transition cost = +∞);
/// this prevents the optimum from landing on an alignment that uses the
/// disallowed bulge type and then gets rejected by the classify step, which
/// would cause us to miss the valid alignment that uses only allowed bulges.
pub fn traceback(
    pattern: &[u8],
    text: &[u8],
    max_edits: u32,
    max_dna_bulges: u32,
    max_rna_bulges: u32,
) -> Option<Alignment> {
    let m = pattern.len();
    let n = text.len();

    const INF: u32 = u32::MAX / 2;
    let dna_cost: u32 = if max_dna_bulges == 0 { INF } else { 1 };
    let rna_cost: u32 = if max_rna_bulges == 0 { INF } else { 1 };

    let mut dp = vec![vec![0u32; n + 1]; m + 1];
    for i in 0..=m { dp[i][0] = (i as u32).saturating_mul(rna_cost).min(INF); }
    // dp[0][j] = 0 (semi-global)

    for i in 1..=m {
        for j in 1..=n {
            let match_cost = if cmp_loose(pattern[i-1], text[j-1]) { 0 } else { 1 };
            dp[i][j] = std::cmp::min(
                dp[i-1][j-1].saturating_add(match_cost),
                std::cmp::min(
                    dp[i-1][j].saturating_add(rna_cost),
                    dp[i][j-1].saturating_add(dna_cost),
                ),
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
            if dp[i][j] == dp[i-1][j-1].saturating_add(match_cost) {
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
            if dp[i][j] == dp[i-1][j].saturating_add(rna_cost) {
                // Deletion: pattern has extra character (RNA bulge)
                ops.push(EditOp::RnaBulge);
                rna_bulges += 1;
                pa.push(pattern[i-1]);
                ta.push(b'-');
                i -= 1;
                continue;
            }
            if dp[i][j] == dp[i][j-1].saturating_add(dna_cost) {
                // Insertion: text has extra character (DNA bulge)
                ops.push(EditOp::DnaBulge);
                dna_bulges += 1;
                pa.push(b'-');
                ta.push(text[j-1]);
                j -= 1;
                continue;
            }
        }
        if i > 0 && (j == 0 || dp[i][j] == dp[i-1][j].saturating_add(rna_cost)) {
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

/// Enumerate ALL distinct alignments of `pattern` ending at `text.len()` that
/// satisfy the per-type caps (mismatches, dna_bulges, rna_bulges).
///
/// Unlike [`traceback`] which returns one optimum, this walks every valid path
/// through the DP. Callers receive one [`Alignment`] per distinct gap placement,
/// matching the original cas-offinder behavior where different bulge
/// placements at the same end position are reported as separate hits.
///
/// `pam_is_n[k]` marks pattern position k as a PAM cell; bulges adjacent to
/// such cells are pruned (cas-offinder convention: no gaps in the PAM region).
///
/// Performance: the DP is filled once (O(m·n)) and then the DFS is pruned by
/// (a) per-type caps and (b) `dp[i][j]` lower bound on remaining cost. In
/// practice this emits only a handful of alignments per candidate.
pub fn traceback_all(
    pattern_bit4: &[u8],
    text_bit4: &[u8],
    max_edits: u32,
    max_dna_bulges: u32,
    max_rna_bulges: u32,
    max_mismatches: u32,
    pam_is_n: &[bool],
) -> Vec<Alignment> {
    let m = pattern_bit4.len();
    let n = text_bit4.len();
    debug_assert_eq!(pam_is_n.len(), m);

    const INF: u32 = u32::MAX / 2;
    let dna_cost: u32 = if max_dna_bulges == 0 { INF } else { 1 };
    let rna_cost: u32 = if max_rna_bulges == 0 { INF } else { 1 };

    // Fill DP using bit4 AND for the match test. Bit4 = 0 is padding at a
    // chromosome boundary and must never match anything; we mark those cells
    // as INF so multi-path enumeration can't straddle two concatenated
    // chromosomes. This is the same comparison the GPU kernel uses so CPU
    // and GPU enumerate the same alignment set.
    let mut dp = vec![vec![0u32; n + 1]; m + 1];
    for i in 0..=m {
        dp[i][0] = (i as u32).saturating_mul(rna_cost).min(INF);
    }
    for i in 1..=m {
        for j in 1..=n {
            if text_bit4[j - 1] == 0 {
                dp[i][j] = INF;
                continue;
            }
            let match_cost = if cmp_loose_bit4(pattern_bit4[i - 1], text_bit4[j - 1]) { 0 } else { 1 };
            dp[i][j] = std::cmp::min(
                dp[i - 1][j - 1].saturating_add(match_cost),
                std::cmp::min(
                    dp[i - 1][j].saturating_add(rna_cost),
                    dp[i][j - 1].saturating_add(dna_cost),
                ),
            );
        }
    }
    if dp[m][n] > max_edits {
        return Vec::new();
    }

    let mut results: Vec<Alignment> = Vec::new();
    let mut ops: Vec<EditOp> = Vec::new();
    enumerate(
        m, n,
        &mut ops,
        0, 0, 0, 0,
        &dp, pattern_bit4, text_bit4,
        max_edits, max_dna_bulges, max_rna_bulges, max_mismatches,
        pam_is_n,
        &mut results,
    );
    results
}

#[allow(clippy::too_many_arguments)]
fn enumerate(
    i: usize,
    j: usize,
    ops: &mut Vec<EditOp>,
    cost: u32,
    mm: u32,
    db: u32,
    rb: u32,
    dp: &[Vec<u32>],
    pattern_bit4: &[u8],
    text_bit4: &[u8],
    max_edits: u32,
    max_dna: u32,
    max_rna: u32,
    max_mm: u32,
    pam_is_n: &[bool],
    results: &mut Vec<Alignment>,
) {
    // Prune: any path through (i,j) has total cost >= cost + dp[i][j].
    if cost.saturating_add(dp[i][j]) > max_edits {
        return;
    }
    if mm > max_mm || db > max_dna || rb > max_rna {
        return;
    }
    if i == 0 {
        // Reached pattern start; semi-global text prefix is free.
        // ops was pushed in walk-back order; reverse for output.
        // Emit pattern_aligned / text_aligned as ASCII (bit4 -> char) so
        // downstream consumers and the output writer can use them directly.
        use crate::bit4ops::bit4_to_char;
        let mut ops_fwd = ops.clone();
        ops_fwd.reverse();
        let mut pa: Vec<u8> = Vec::with_capacity(ops_fwd.len());
        let mut ta: Vec<u8> = Vec::with_capacity(ops_fwd.len());
        let mut pi = 0usize;
        let mut ti = j;
        for op in &ops_fwd {
            match op {
                EditOp::Match | EditOp::Substitution => {
                    pa.push(bit4_to_char(pattern_bit4[pi]));
                    ta.push(bit4_to_char(text_bit4[ti]));
                    pi += 1;
                    ti += 1;
                }
                EditOp::RnaBulge => {
                    pa.push(bit4_to_char(pattern_bit4[pi]));
                    ta.push(b'-');
                    pi += 1;
                }
                EditOp::DnaBulge => {
                    pa.push(b'-');
                    ta.push(bit4_to_char(text_bit4[ti]));
                    ti += 1;
                }
            }
        }
        results.push(Alignment {
            ops: ops_fwd,
            pattern_aligned: pa,
            text_aligned: ta,
            text_start: j,
            mismatches: mm,
            dna_bulges: db,
            rna_bulges: rb,
        });
        return;
    }

    // Diagonal: match / substitution
    if j > 0 {
        let mc = if cmp_loose_bit4(pattern_bit4[i - 1], text_bit4[j - 1]) { 0 } else { 1 };
        let touches_n = mc == 1 && pam_is_n[i - 1];
        if !touches_n {
            ops.push(if mc == 0 { EditOp::Match } else { EditOp::Substitution });
            enumerate(
                i - 1, j - 1, ops,
                cost + mc, mm + mc, db, rb,
                dp, pattern_bit4, text_bit4,
                max_edits, max_dna, max_rna, max_mm,
                pam_is_n, results,
            );
            ops.pop();
        }
    }

    // RNA bulge: pattern[i-1] consumed, no text char.
    if rb < max_rna && max_rna > 0 {
        let pattern_pos_is_n = pam_is_n[i - 1];
        if !pattern_pos_is_n {
            ops.push(EditOp::RnaBulge);
            enumerate(
                i - 1, j, ops,
                cost + 1, mm, db, rb + 1,
                dp, pattern_bit4, text_bit4,
                max_edits, max_dna, max_rna, max_mm,
                pam_is_n, results,
            );
            ops.pop();
        }
    }

    // DNA bulge: text char consumed, no pattern char.
    //
    // C++ cas-offinder-bulge allows a bulge between pattern positions i-1 and
    // i iff:
    //   * The bulge is NOT strictly inside the PAM (i.e. not both neighbours
    //     are N). Bulges at the crRNA/PAM boundary — where one neighbour is
    //     crRNA and the other is the first/last PAM N — are allowed.
    //   * The bulge does NOT extend past the PAM side of the pattern. In
    //     backward-DFS terms this is `i == pattern_len` (past-the-end). The
    //     other end `i == 0` is already unreachable here because enumerate()
    //     emits and returns as soon as i hits 0.
    if db < max_dna && max_dna > 0 && j > 0 && i < pattern_bit4.len() {
        let prev_n = i > 0 && pam_is_n[i - 1];
        let next_n = pam_is_n[i];
        if !(prev_n && next_n) {
            ops.push(EditOp::DnaBulge);
            enumerate(
                i, j - 1, ops,
                cost + 1, mm, db + 1, rb,
                dp, pattern_bit4, text_bit4,
                max_edits, max_dna, max_rna, max_mm,
                pam_is_n, results,
            );
            ops.pop();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_traceback_exact() {
        let a = traceback(b"ACGT", b"ACGT", 2, 2, 2).unwrap();
        assert_eq!(a.mismatches, 0);
        assert_eq!(a.dna_bulges, 0);
        assert_eq!(a.rna_bulges, 0);
        assert_eq!(a.pattern_aligned, b"ACGT");
        assert_eq!(a.text_aligned, b"ACGT");
    }

    #[test]
    fn test_traceback_substitution() {
        let a = traceback(b"ACGT", b"AAGT", 2, 2, 2).unwrap();
        assert_eq!(a.mismatches, 1);
        assert_eq!(a.dna_bulges, 0);
        assert_eq!(a.rna_bulges, 0);
    }

    #[test]
    fn test_traceback_dna_bulge() {
        // text has extra X: "ACGT" vs "ACXGT"
        let a = traceback(b"ACGT", b"ACXGT", 2, 2, 2).unwrap();
        assert_eq!(a.dna_bulges, 1);
        assert_eq!(a.mismatches, 0);
        assert_eq!(a.rna_bulges, 0);
        assert_eq!(a.pattern_aligned, b"AC-GT");
        assert_eq!(a.text_aligned, b"ACXGT");
    }

    #[test]
    fn test_traceback_rna_bulge() {
        // pattern has extra char: "ACGT" vs "ACT" (G deleted from text)
        let a = traceback(b"ACGT", b"ACT", 2, 2, 2).unwrap();
        assert_eq!(a.rna_bulges, 1);
        assert_eq!(a.mismatches, 0);
        assert_eq!(a.dna_bulges, 0);
        assert_eq!(a.pattern_aligned, b"ACGT");
        assert_eq!(a.text_aligned, b"AC-T");
    }

    #[test]
    fn test_traceback_exceeds_max() {
        let result = traceback(b"ACGT", b"TTTT", 1, 2, 2);
        assert!(result.is_none(), "Should not find alignment with 1 edit");
    }

    #[test]
    fn test_traceback_forbids_rna_when_cap_zero() {
        // Pattern longer than text forces RNA bulges in semi-global DP (text
        // end is fixed). With max_rna_bulges = 0, traceback must return None.
        let r = traceback(b"ACGTA", b"ACT", 3, 2, 0);
        assert!(r.is_none(), "must not use RNA bulges when forbidden");
    }
}
