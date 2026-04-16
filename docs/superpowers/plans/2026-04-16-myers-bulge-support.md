# Myers Bit-Parallel Bulge Support Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add DNA/RNA bulge search to cas-offinder-rust by replacing popcount-based brute force with PAM scan + Myers bit-parallel edit distance algorithm.

**Architecture:** Two-phase algorithm fused in a single sweep: (1) scan each genome position for PAM exact match, (2) if PAM matches, run Myers bit-parallel edit distance on the guide portion. Traceback reconstructs alignment to classify each match as substitution/DNA-bulge/RNA-bulge. Output format matches cas-offinder-bulge, with a new `.log` file written alongside.

**Tech Stack:**
- Rust (cas-offinder-lib, cas-offinder-cli)
- OpenCL 4bit-encoded genome, pattern
- Myers 1999 bit-parallel edit distance (fits in u64 for patterns ≤64nt)
- Existing chunked genome streaming pipeline

**Build note:** All `cargo build` commands in this plan must be run with `RUSTFLAGS="-L/bce/groups/pnucolab/analysis/cas-offinder-rust/cas-offinder-cli"` to find the OpenCL symlink. Wrap as: `RUSTFLAGS="-L$(pwd)/../cas-offinder-cli" cargo build --release` when invoked from `cas-offinder-cli/` directory.

**Spec reference:** `docs/superpowers/specs/2026-04-16-myers-bulge-design.md`

---

## File Structure

**New files:**
- `cas-offinder-lib/src/myers.rs` — Myers bit-parallel algorithm (CPU)
- `cas-offinder-lib/src/traceback.rs` — Alignment reconstruction for bulge type classification
- `cas-offinder-lib/src/log_writer.rs` — Log file generation
- `cas-offinder-lib/src/kernel_myers.cl` — OpenCL kernel with Myers + PAM check

**Modified files:**
- `cas-offinder-lib/src/lib.rs` — Export new modules
- `cas-offinder-lib/src/search.rs` — Replace `search_chunk_cpu` and `search_device_ocl` with Myers-based search; update data structures
- `cas-offinder-lib/src/chrom_chunk.rs` — Add bulge info to `Match` struct
- `cas-offinder-cli/src/cli_utils.rs` — Parse bulge parameters from input file
- `cas-offinder-cli/src/main.rs` — New output format, log file generation, pass bulge params to search

**Tests added:**
- `cas-offinder-lib/src/myers.rs` (inline `#[cfg(test)]` module)
- `cas-offinder-lib/src/traceback.rs` (inline tests)
- `cas-offinder-lib/tests/integration_bulge.rs` (integration with test data)

---

## Phase 1: Myers Bit-Parallel Library Function (CPU)

This phase adds a pure library function for Myers edit distance and its tests. Does not integrate with search pipeline yet. After this phase, `cargo test` should pass and the binary should still behave as before (no functional change).

### Task 1.1: Add myers.rs module stub and tests

**Files:**
- Create: `cas-offinder-lib/src/myers.rs`
- Modify: `cas-offinder-lib/src/lib.rs`

- [ ] **Step 1: Create `myers.rs` with module docstring and signature**

Write the file:

```rust
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
            // Ambiguous: match all bases
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

    let last_bit = 1u64 << (m - 1);
    let mut vp: u64 = if m < 64 { (1u64 << m) - 1 } else { !0u64 };
    let mut vn: u64 = 0;
    let mut score: i32 = m as i32;

    for &c in text {
        let idx = nucl_idx(c);
        let eq = if idx < 4 {
            peq.peq[idx as usize]
        } else {
            // Ambiguous text char matches everything
            peq.peq[0] | peq.peq[1] | peq.peq[2] | peq.peq[3]
        };
        let x = eq | vn;
        let d0 = (((x & vp).wrapping_add(vp)) ^ vp) | x;
        let hn = vp & d0;
        let hp = vn | !(vp | d0);
        let x_shift = (hp << 1) | 1;
        vn = x_shift & d0;
        vp = (hn << 1) | !(x_shift | d0);
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
}
```

- [ ] **Step 2: Export from lib.rs**

Modify `cas-offinder-lib/src/lib.rs`, add after existing `mod` lines:

```rust
mod myers;
pub use crate::myers::*;
```

- [ ] **Step 3: Run test to verify module compiles**

Run: `cd cas-offinder-lib && cargo test --lib myers::tests::test_nucl_idx -- --nocapture`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add cas-offinder-lib/src/myers.rs cas-offinder-lib/src/lib.rs
git commit -m "feat(myers): add skeleton module with Peq builder and edit distance"
```

### Task 1.2: Test Myers on exact-match cases (zero edit distance)

**Files:**
- Modify: `cas-offinder-lib/src/myers.rs` (add tests)

- [ ] **Step 1: Write tests for exact match cases**

Add to `#[cfg(test)] mod tests` block:

```rust
#[test]
fn test_myers_exact_match() {
    let pattern = b"ACGT";
    let peq = build_peq(pattern);
    // Text is exactly the pattern
    assert_eq!(myers_edit_distance(&peq, b"ACGT"), 0);
}

#[test]
fn test_myers_exact_match_at_end() {
    let pattern = b"ACGT";
    let peq = build_peq(pattern);
    // Pattern appears at end of text (semi-global: free start gap)
    assert_eq!(myers_edit_distance(&peq, b"TTACGT"), 0);
}

#[test]
fn test_myers_exact_match_with_n_text() {
    let pattern = b"ACGT";
    let peq = build_peq(pattern);
    // N in text matches anything
    assert_eq!(myers_edit_distance(&peq, b"ACGN"), 0);
}
```

- [ ] **Step 2: Run tests**

Run: `cd cas-offinder-lib && cargo test --lib myers::tests -- --nocapture`
Expected: PASS (all 4 tests including nucl_idx)

- [ ] **Step 3: Commit**

```bash
git add cas-offinder-lib/src/myers.rs
git commit -m "test(myers): add exact-match tests"
```

### Task 1.3: Test Myers on substitution (mismatch only)

**Files:**
- Modify: `cas-offinder-lib/src/myers.rs`

- [ ] **Step 1: Write tests for substitution cases**

Add to tests module:

```rust
#[test]
fn test_myers_one_substitution() {
    let pattern = b"ACGT";
    let peq = build_peq(pattern);
    // One substitution: C -> A
    assert_eq!(myers_edit_distance(&peq, b"AAGT"), 1);
    // One substitution: G -> T
    assert_eq!(myers_edit_distance(&peq, b"ACTT"), 1);
}

#[test]
fn test_myers_multiple_substitutions() {
    let pattern = b"ACGT";
    let peq = build_peq(pattern);
    // Two substitutions
    assert_eq!(myers_edit_distance(&peq, b"AAGA"), 2);
    // All different (one match at position 0)
    assert_eq!(myers_edit_distance(&peq, b"ATTC"), 3);
}

#[test]
fn test_myers_long_pattern_substitutions() {
    let pattern = b"ACGTACGTACGTACGTACGT"; // 20nt
    let peq = build_peq(pattern);
    assert_eq!(myers_edit_distance(&peq, b"ACGTACGTACGTACGTACGT"), 0);
    assert_eq!(myers_edit_distance(&peq, b"ACGTAAGTACGTACGTACGT"), 1); // position 5: C->A
    assert_eq!(myers_edit_distance(&peq, b"ACGTACGTACGTACGTACGA"), 1); // last T->A
}
```

- [ ] **Step 2: Run tests**

Run: `cd cas-offinder-lib && cargo test --lib myers::tests -- --nocapture`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add cas-offinder-lib/src/myers.rs
git commit -m "test(myers): add substitution tests"
```

### Task 1.4: Test Myers on DNA bulges (insertion in genome/text)

**Files:**
- Modify: `cas-offinder-lib/src/myers.rs`

- [ ] **Step 1: Write tests for DNA bulges**

DNA bulge = extra character in text (genome) relative to pattern. Pattern is 20nt, text has 21nt with one extra base inserted somewhere.

Add tests:

```rust
#[test]
fn test_myers_one_dna_bulge() {
    let pattern = b"ACGTACGT";
    let peq = build_peq(pattern);
    // Text has extra 'T' inserted at position 4 → edit distance 1 (one insertion in text)
    assert_eq!(myers_edit_distance(&peq, b"ACGTTACGT"), 1);
    // Extra base at beginning doesn't count (semi-global)
    assert_eq!(myers_edit_distance(&peq, b"TACGTACGT"), 0);
}

#[test]
fn test_myers_two_dna_bulges() {
    let pattern = b"ACGTACGT";
    let peq = build_peq(pattern);
    // Two insertions in text
    assert_eq!(myers_edit_distance(&peq, b"ACGNTAACGT"), 2);
}
```

- [ ] **Step 2: Run tests**

Run: `cd cas-offinder-lib && cargo test --lib myers::tests -- --nocapture`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add cas-offinder-lib/src/myers.rs
git commit -m "test(myers): add DNA bulge tests"
```

### Task 1.5: Test Myers on RNA bulges (deletion in genome/text)

**Files:**
- Modify: `cas-offinder-lib/src/myers.rs`

- [ ] **Step 1: Write tests for RNA bulges**

RNA bulge = missing character in text relative to pattern (pattern has extra base).

Add tests:

```rust
#[test]
fn test_myers_one_rna_bulge() {
    let pattern = b"ACGTACGT";
    let peq = build_peq(pattern);
    // Text is shorter by 1: deleted the 'T' at position 3 of pattern
    assert_eq!(myers_edit_distance(&peq, b"ACGACGT"), 1);
}

#[test]
fn test_myers_two_rna_bulges() {
    let pattern = b"ACGTACGT";
    let peq = build_peq(pattern);
    // Missing two characters (one from middle, one from end)
    assert_eq!(myers_edit_distance(&peq, b"ACGACG"), 2);
}
```

- [ ] **Step 2: Run tests**

Run: `cd cas-offinder-lib && cargo test --lib myers::tests -- --nocapture`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add cas-offinder-lib/src/myers.rs
git commit -m "test(myers): add RNA bulge tests"
```

### Task 1.6: Test Myers on mixed edits (mismatch + bulge)

**Files:**
- Modify: `cas-offinder-lib/src/myers.rs`

- [ ] **Step 1: Write tests for mixed edit cases**

```rust
#[test]
fn test_myers_mixed_edits() {
    let pattern = b"ACGTACGT";
    let peq = build_peq(pattern);
    // One mismatch + one DNA bulge
    // pattern:  A C G T A C G T
    // text:     A C G T X A A G T  (X insertion, C→A mismatch)
    assert_eq!(myers_edit_distance(&peq, b"ACGTXAAGT"), 2);
}

#[test]
fn test_myers_worst_case() {
    let pattern = b"ACGTACGT";
    let peq = build_peq(pattern);
    // Completely unrelated
    let result = myers_edit_distance(&peq, b"TTTTTTTT");
    assert!(result >= 4 && result <= 8, "result was {}", result);
}
```

- [ ] **Step 2: Run tests**

Run: `cd cas-offinder-lib && cargo test --lib myers::tests -- --nocapture`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add cas-offinder-lib/src/myers.rs
git commit -m "test(myers): add mixed edit tests"
```

### Task 1.7: Add reference DP implementation for cross-validation

**Files:**
- Modify: `cas-offinder-lib/src/myers.rs`

- [ ] **Step 1: Add a slow-but-obviously-correct DP reference implementation in tests**

Add in `#[cfg(test)]` block:

```rust
// Reference implementation: standard edit distance DP (semi-global).
// Slow O(m*n) but obviously correct. Only used to cross-check Myers.
fn reference_edit_distance(pattern: &[u8], text: &[u8]) -> u32 {
    let m = pattern.len();
    let n = text.len();
    // dp[i][j] = min edit distance to match pattern[0..i] against any suffix of text[0..j]
    let mut dp = vec![vec![0u32; n + 1]; m + 1];
    for i in 0..=m { dp[i][0] = i as u32; }
    // dp[0][j] = 0 (semi-global: free start)
    for i in 1..=m {
        for j in 1..=n {
            let match_cost = if cmp_char_loose(pattern[i-1], text[j-1]) { 0 } else { 1 };
            dp[i][j] = std::cmp::min(
                dp[i-1][j-1] + match_cost,
                std::cmp::min(
                    dp[i-1][j] + 1,   // deletion (RNA bulge)
                    dp[i][j-1] + 1,   // insertion (DNA bulge)
                ),
            );
        }
    }
    // Final: min edit distance with pattern fully matched ending at any text position
    // For Myers we track the last text position specifically; reference matches that
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
        let m = 8 + (seed % 17) as usize; // 8..24
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
            "pattern={:?} text={:?}", std::str::from_utf8(&pattern).unwrap(),
            std::str::from_utf8(&text).unwrap());
    }
}
```

- [ ] **Step 2: Run cross-validation**

Run: `cd cas-offinder-lib && cargo test --lib myers::tests::test_myers_matches_reference_random -- --nocapture`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add cas-offinder-lib/src/myers.rs
git commit -m "test(myers): add randomized cross-validation against reference DP"
```

---

## Phase 2: Traceback for Bulge Classification

To distinguish DNA bulge from RNA bulge from substitution, we need to reconstruct the alignment at match positions. Myers only gives us the edit distance; traceback gives us the edit *types*.

### Task 2.1: Add traceback module with DP-based reconstruction

**Files:**
- Create: `cas-offinder-lib/src/traceback.rs`
- Modify: `cas-offinder-lib/src/lib.rs`

- [ ] **Step 1: Create `traceback.rs`**

```rust
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
```

- [ ] **Step 2: Export from lib.rs**

Add to `cas-offinder-lib/src/lib.rs`:

```rust
mod traceback;
pub use crate::traceback::*;
```

- [ ] **Step 3: Run tests**

Run: `cd cas-offinder-lib && cargo test --lib traceback::tests -- --nocapture`
Expected: PASS (5 tests)

- [ ] **Step 4: Commit**

```bash
git add cas-offinder-lib/src/traceback.rs cas-offinder-lib/src/lib.rs
git commit -m "feat(traceback): add alignment reconstruction for bulge classification"
```

---

## Phase 3: Input Format Update (Bulge Parameters)

Parse bulge_dna and bulge_rna from the 2nd line of the input file. Keep backward-compatible with current format (no bulge params = 0 bulges).

### Task 3.1: Extend SearchRunInfo with bulge parameters

**Files:**
- Modify: `cas-offinder-cli/src/cli_utils.rs`

- [ ] **Step 1: Add bulge_dna and bulge_rna fields to SearchRunInfo**

Change the struct in `cas-offinder-cli/src/cli_utils.rs`:

```rust
pub struct SearchRunInfo {
    pub genome_path: String,
    pub out_path: String,
    pub dev_ty: OclDeviceType,
    pub search_filter: Vec<u8>,
    pub patterns: Vec<Vec<u8>>,
    pub pattern_infos: Vec<String>,
    pub pattern_len: usize,
    pub max_mismatches: u32,
    pub max_dna_bulges: u32,  // NEW
    pub max_rna_bulges: u32,  // NEW
}
```

And `InFileInfo`:

```rust
struct InFileInfo {
    genome_path: String,
    search_filter: Vec<u8>,
    patterns: Vec<Vec<u8>>,
    pattern_infos: Vec<String>,
    pattern_len: usize,
    max_mismatches: u32,
    max_dna_bulges: u32,
    max_rna_bulges: u32,
}
```

- [ ] **Step 2: Update parse_and_validate_input to parse bulge params**

Modify the parsing block that reads `searcher_line`. Replace:

```rust
    let searcher_line = line_iter
        .next()
        .ok_or(CliError::ArgumentError(file_too_short_err))??;
    let search_filter = searcher_line.trim_end().as_bytes().to_vec();
    if !is_mixedbase_str(&search_filter) {
        return Err(mixed_base_error);
    }
    let pattern_len = search_filter.len();
```

With:

```rust
    let searcher_line = line_iter
        .next()
        .ok_or(CliError::ArgumentError(file_too_short_err))??;
    let parts: Vec<&str> = searcher_line.split_ascii_whitespace().collect();
    let (search_filter, max_dna_bulges, max_rna_bulges) = match parts.len() {
        1 => (parts[0].as_bytes().to_vec(), 0u32, 0u32),
        3 => {
            let dna = parts[1].parse::<u32>().map_err(|_|
                CliError::ArgumentError("bulge_dna must be a non-negative integer"))?;
            let rna = parts[2].parse::<u32>().map_err(|_|
                CliError::ArgumentError("bulge_rna must be a non-negative integer"))?;
            (parts[0].as_bytes().to_vec(), dna, rna)
        },
        _ => return Err(CliError::ArgumentError(
            "2nd line must be: <search_filter> [<bulge_dna> <bulge_rna>]")),
    };
    if !is_mixedbase_str(&search_filter) {
        return Err(mixed_base_error);
    }
    let pattern_len = search_filter.len();
```

Also update the return value at the bottom of `parse_and_validate_input`. Change:

```rust
        Some(max_mismatches) => Ok(InFileInfo {
            genome_path,
            search_filter,
            patterns,
            pattern_infos,
            pattern_len,
            max_mismatches,
        }),
```

To:

```rust
        Some(max_mismatches) => Ok(InFileInfo {
            genome_path,
            search_filter,
            patterns,
            pattern_infos,
            pattern_len,
            max_mismatches,
            max_dna_bulges,
            max_rna_bulges,
        }),
```

And update `parse_and_validate_args` to copy these fields to `SearchRunInfo`:

```rust
    Ok(SearchRunInfo {
        genome_path: parsed_in_file.genome_path,
        search_filter: parsed_in_file.search_filter,
        patterns: parsed_in_file.patterns,
        pattern_infos: parsed_in_file.pattern_infos,
        pattern_len: parsed_in_file.pattern_len,
        max_mismatches: parsed_in_file.max_mismatches,
        max_dna_bulges: parsed_in_file.max_dna_bulges,
        max_rna_bulges: parsed_in_file.max_rna_bulges,
        out_path: out_filename.clone(),
        dev_ty: get_dev_ty(device_ty_str)?,
    })
```

- [ ] **Step 3: Add a test for parsing bulge params**

Add to the `#[cfg(test)] mod tests` at bottom of `cli_utils.rs`:

```rust
    use std::io::Write;

    fn write_temp_input(content: &str) -> String {
        let path = format!("/tmp/test_input_{}.in", std::process::id());
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(content.as_bytes()).unwrap();
        path
    }

    #[test]
    fn test_parse_input_no_bulge() {
        let input = "/tmp/genome.fa\nNNNNNNNNNNNNNNNNNNNNNNN\nACGTACGTACGTACGTACGTACG 5\n";
        let path = write_temp_input(input);
        let info = parse_and_validate_input(&path).unwrap();
        assert_eq!(info.max_dna_bulges, 0);
        assert_eq!(info.max_rna_bulges, 0);
        assert_eq!(info.max_mismatches, 5);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_parse_input_with_bulge() {
        let input = "/tmp/genome.fa\nNNNNNNNNNNNNNNNNNNNNNRG 2 1\nACGTACGTACGTACGTACGTA 5\n";
        let path = write_temp_input(input);
        let info = parse_and_validate_input(&path).unwrap();
        assert_eq!(info.max_dna_bulges, 2);
        assert_eq!(info.max_rna_bulges, 1);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_parse_input_invalid_bulge_count() {
        let input = "/tmp/genome.fa\nNNN 2\nACGT 5\n";
        let path = write_temp_input(input);
        let result = parse_and_validate_input(&path);
        assert!(result.is_err());
        std::fs::remove_file(&path).ok();
    }
```

Also mark `parse_and_validate_input` as `pub(crate)` (or `pub` in the module) for tests to access it.

Change the function signature:

```rust
pub fn parse_and_validate_input(in_path: &String) -> Result<InFileInfo> {
```

And make `InFileInfo` pub(crate) so tests can access:

```rust
pub(crate) struct InFileInfo {
    // ...
}
```

Actually — a simpler approach: make the fields on `SearchRunInfo` public (they already are), and only test via `parse_and_validate_args`. Revert the pub(crate) changes if easier.

- [ ] **Step 4: Run tests**

Run: `cd cas-offinder-cli && RUSTFLAGS="-L$(pwd)" cargo test --bin cas-offinder-cli -- --nocapture`
Expected: PASS (all existing tests + new 3)

- [ ] **Step 5: Commit**

```bash
git add cas-offinder-cli/src/cli_utils.rs
git commit -m "feat(cli): parse optional bulge_dna and bulge_rna from 2nd line"
```

---

## Phase 4: Match Struct Update (Bulge Info)

Add bulge information to the `Match` struct in the library so downstream code can use it.

### Task 4.1: Extend Match struct with bulge fields

**Files:**
- Modify: `cas-offinder-lib/src/chrom_chunk.rs`

- [ ] **Step 1: Inspect existing Match struct**

Run: `cd cas-offinder-lib && grep -n 'pub struct Match' src/chrom_chunk.rs`
Expected output: shows current Match struct definition

- [ ] **Step 2: Add new fields**

Modify the `Match` struct. Change:

```rust
pub struct Match {
    pub chr_name: String,
    pub chrom_idx: u64,
    pub pattern_idx: u32,
    pub mismatches: u32,
    pub is_forward: bool,
    pub dna_seq: Vec<u8>,
    pub rna_seq: Vec<u8>,
}
```

To:

```rust
pub struct Match {
    pub chr_name: String,
    pub chrom_idx: u64,
    pub pattern_idx: u32,
    pub mismatches: u32,
    pub is_forward: bool,
    pub dna_seq: Vec<u8>,   // with '-' for gaps in DNA
    pub rna_seq: Vec<u8>,   // with '-' for gaps in RNA
    pub dna_bulge_size: u32, // number of gap characters on DNA side (== RNA bulge on pattern side)
    pub rna_bulge_size: u32, // number of gap characters on RNA side (== DNA bulge on genome side)
}
```

**Note on naming**: the field naming here follows `cas-offinder-bulge` convention where "RNA bulge" means pattern has extra (genome shorter, hence gap shown in DNA sequence), and "DNA bulge" means genome has extra (pattern has gap). So:
- `rna_bulge_size` = gaps appearing in RNA/pattern display
- `dna_bulge_size` = gaps appearing in DNA/genome display

Actually wait — re-check cas-offinder-bulge Python output:
- Type "DNA": genome has extra chars → crRNA shows gap
- Type "RNA": pattern has extra chars → DNA shows gap

So naming should be semantic:
- `dna_bulge_size` = DNA has extra (pattern display shows `-`)
- `rna_bulge_size` = RNA has extra (DNA display shows `-`)

Adjust comment:

```rust
    pub dna_bulge_size: u32, // number of extra chars in DNA (crRNA shown with '-')
    pub rna_bulge_size: u32, // number of extra chars in RNA (DNA shown with '-')
```

- [ ] **Step 3: Find all constructors of Match and update them**

Run: `cd cas-offinder-lib && grep -n 'Match {' src/`
Expected: shows `src/search.rs:424-433` where `convert_matches` constructs `Match`.

- [ ] **Step 4: Update convert_matches to set bulge fields to 0**

In `src/search.rs`, the `convert_matches` function. Find the block:

```rust
            results.push(Match {
                chr_name: search_res.meta.chr_names[idx].clone(),
                chrom_idx: pos,
                pattern_idx: smatch.pattern_idx,
                mismatches: smatch.mismatches,
                is_forward: is_forward,
                dna_seq: dna_result,
                rna_seq: rna_result,
            });
```

Update to:

```rust
            results.push(Match {
                chr_name: search_res.meta.chr_names[idx].clone(),
                chrom_idx: pos,
                pattern_idx: smatch.pattern_idx,
                mismatches: smatch.mismatches,
                is_forward: is_forward,
                dna_seq: dna_result,
                rna_seq: rna_result,
                dna_bulge_size: 0,
                rna_bulge_size: 0,
            });
```

- [ ] **Step 5: Build to check everything compiles**

Run: `cd cas-offinder-cli && RUSTFLAGS="-L$(pwd)" cargo build 2>&1 | tail -10`
Expected: builds successfully

- [ ] **Step 6: Run existing tests to verify no regression**

Run: `cd cas-offinder-cli && RUSTFLAGS="-L$(pwd)" cargo test 2>&1 | tail -20`
Expected: all existing tests pass

- [ ] **Step 7: Commit**

```bash
git add cas-offinder-lib/src/chrom_chunk.rs cas-offinder-lib/src/search.rs
git commit -m "feat(match): add dna_bulge_size and rna_bulge_size fields to Match"
```

---

## Phase 5: Core Integration (CPU Only) — Guide Extraction + Myers Search

Replace the CPU search path with Myers-based PAM scan + edit distance. Keep GPU path on the old algorithm for now (will be replaced in Phase 7).

### Task 5.1: Add helper to extract guide region from N-padded pattern

**Files:**
- Modify: `cas-offinder-lib/src/bit4ops.rs`

- [ ] **Step 1: Add helper functions**

Add at the bottom of `cas-offinder-lib/src/bit4ops.rs`, before the test module:

```rust
/// Determine PAM orientation from search_filter. Returns (is_reversed, pam_len).
/// A "normal" filter has N-block on the left (guide) and actual bases on right (PAM).
/// A "reversed" filter has N-block on the right (guide) and actual bases on left (PAM).
pub fn detect_pam_orientation(search_filter: &[u8]) -> (bool, usize) {
    let n = search_filter.len();
    let mut left_ns = 0;
    for &c in search_filter {
        if c == b'N' || c == b'n' { left_ns += 1; } else { break; }
    }
    let mut right_ns = 0;
    for &c in search_filter.iter().rev() {
        if c == b'N' || c == b'n' { right_ns += 1; } else { break; }
    }
    if left_ns >= right_ns {
        // N block on left → PAM on right (normal SpCas9 NGG)
        (false, n - left_ns)
    } else {
        (true, n - right_ns)
    }
}

/// Extract the guide portion of a pattern given PAM orientation.
/// For non-reversed: strip trailing Ns (PAM placeholder).
/// For reversed: strip leading Ns.
pub fn extract_guide(pattern: &[u8], is_reversed: bool, pam_len: usize) -> &[u8] {
    let n = pattern.len();
    if is_reversed {
        &pattern[pam_len..]
    } else {
        &pattern[..n - pam_len]
    }
}
```

- [ ] **Step 2: Add tests**

Inside the `#[cfg(test)] mod tests`:

```rust
    #[test]
    fn test_detect_pam_orientation_normal() {
        let filter = b"NNNNNNNNNNNNNNNNNNNNNNGG";
        let (reversed, pam_len) = detect_pam_orientation(filter);
        assert_eq!(reversed, false);
        assert_eq!(pam_len, 2);
    }

    #[test]
    fn test_detect_pam_orientation_reversed() {
        let filter = b"TTTNNNNNNNNNNNNNNNNNNNNN";
        let (reversed, pam_len) = detect_pam_orientation(filter);
        assert_eq!(reversed, true);
        assert_eq!(pam_len, 3);
    }

    #[test]
    fn test_extract_guide_normal() {
        let pattern = b"ACGTACGTACGTACGTACGTNNN";
        let guide = extract_guide(pattern, false, 3);
        assert_eq!(guide, b"ACGTACGTACGTACGTACGT");
    }

    #[test]
    fn test_extract_guide_reversed() {
        let pattern = b"NNNACGTACGTACGTACGTACGT";
        let guide = extract_guide(pattern, true, 3);
        assert_eq!(guide, b"ACGTACGTACGTACGTACGT");
    }
```

- [ ] **Step 3: Run tests**

Run: `cd cas-offinder-lib && cargo test --lib bit4ops::tests -- --nocapture`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add cas-offinder-lib/src/bit4ops.rs
git commit -m "feat(bit4ops): add PAM orientation detection and guide extraction"
```

### Task 5.2: Define search_chunk_myers function signature and unit tests

**Files:**
- Modify: `cas-offinder-lib/src/search.rs`

- [ ] **Step 1: Add new fields to SearchMatch to carry bulge info**

Find the `SearchMatch` struct in `src/search.rs` and update:

```rust
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct SearchMatch {
    pub chunk_idx: u32,
    pub pattern_idx: u32,
    pub mismatches: u32,
    // NEW:
    pub dna_bulge_size: u16,   // extra chars in DNA (pattern shorter)
    pub rna_bulge_size: u16,   // extra chars in pattern (DNA shorter)
}
```

Important: `#[repr(C)]` layout must stay compatible with OpenCL struct. Size increase: 12 → 16 bytes (4-byte aligned). This requires updating the OpenCL kernel too (Task 7.1), but for now we're only affecting CPU code.

Update any default constructions. Search for `SearchMatch {` in the file and ensure all places set the new fields.

Run: `cd cas-offinder-lib && grep -n 'SearchMatch {' src/search.rs`

Update each match. Common pattern:

```rust
SearchMatch { chunk_idx: 0, pattern_idx: 0, mismatches: 0 }
```

to:

```rust
SearchMatch { chunk_idx: 0, pattern_idx: 0, mismatches: 0, dna_bulge_size: 0, rna_bulge_size: 0 }
```

- [ ] **Step 2: Build**

Run: `cd cas-offinder-cli && RUSTFLAGS="-L$(pwd)" cargo build 2>&1 | tail -5`
Expected: builds

- [ ] **Step 3: Commit**

```bash
git add cas-offinder-lib/src/search.rs
git commit -m "feat(search): extend SearchMatch with bulge size fields"
```

### Task 5.3: Implement search_chunk_myers (CPU path with Myers)

**Files:**
- Modify: `cas-offinder-lib/src/search.rs`

- [ ] **Step 1: Add new search function**

Add (do not yet replace `search_chunk_cpu`) a new function in `src/search.rs`:

```rust
use crate::myers::{build_peq, myers_edit_distance, PeqTable};
use crate::traceback::traceback;

fn search_chunk_myers(
    max_mismatches: u32,
    max_dna_bulges: u32,
    max_rna_bulges: u32,
    search_filter: &[u8],
    pam_is_reversed: bool,
    pam_len: usize,
    guide_len: usize,
    guide_patterns_ascii: &[Vec<u8>],
    chunk_data_bit4: &[u8; SEARCH_CHUNK_SIZE_BYTES],
) -> Vec<SearchMatch> {
    let max_edits = max_mismatches + max_dna_bulges + max_rna_bulges;
    let n_patterns = guide_patterns_ascii.len();
    let total_nucl = SEARCH_CHUNK_SIZE_BYTES * 2;

    // Decode chunk to ASCII for Myers (simple approach; optimize later if needed)
    let mut ascii_chunk = vec![0u8; total_nucl];
    crate::bit4ops::bit4_to_string(&mut ascii_chunk, chunk_data_bit4, 0, total_nucl);

    // Pre-build Peq tables
    let peqs: Vec<PeqTable> = guide_patterns_ascii.iter().map(|p| build_peq(p)).collect();

    // Pre-build PAM filter bit-pattern (ASCII form for comparison)
    let pam_filter: Vec<u8> = if pam_is_reversed {
        search_filter[..pam_len].to_vec()
    } else {
        search_filter[(search_filter.len() - pam_len)..].to_vec()
    };

    let mut matches: Vec<SearchMatch> = Vec::new();

    // Total window size needed for Myers: guide_len + max_dna_bulges
    let myers_text_len = guide_len + max_dna_bulges as usize;

    // For each position: check PAM, then Myers
    // Pattern layout (non-reversed): [guide (guide_len)] [PAM (pam_len)]
    // PAM position in genome: starts at j + guide_len
    // Guide text window: genome[j - max_dna_bulges .. j + guide_len]
    //
    // We iterate j such that PAM fits: j + guide_len + pam_len <= total_nucl
    // And guide text window fits: j >= 0 (ensured by skipping early positions)

    // Range of valid start positions:
    let min_j: usize = 0;
    let max_j: usize = total_nucl.saturating_sub(guide_len + pam_len);

    for j in min_j..=max_j {
        // Phase 1: PAM check
        let pam_start = if pam_is_reversed { j } else { j + guide_len };
        if !pam_matches(&ascii_chunk[pam_start..pam_start + pam_len], &pam_filter) {
            continue;
        }

        // Phase 2: Myers on guide
        // Guide text window: span the possible alignments given max_dna_bulges
        // Non-reversed: guide is before PAM.
        //   Text window end = pam_start (= j + guide_len)
        //   Text window start = pam_start - myers_text_len (bounded >= 0)
        let (win_start, win_end) = if pam_is_reversed {
            // Reversed: guide is after PAM.
            //   Text window start = pam_start + pam_len = j + pam_len
            //   Text window end = win_start + myers_text_len
            let s = j + pam_len;
            let e = (s + myers_text_len).min(total_nucl);
            (s, e)
        } else {
            let e = pam_start;
            let s = e.saturating_sub(myers_text_len);
            (s, e)
        };
        let text_window = &ascii_chunk[win_start..win_end];

        for (p_idx, peq) in peqs.iter().enumerate() {
            let edit_dist = if pam_is_reversed {
                // For reversed, we align pattern from left of text
                // Myers in its standard form does semi-global with free start on text
                // For reversed alignment we reverse both pattern and text
                // Simpler: skip for now, handled in pattern list symmetry
                // We expect caller to pass reverse-complemented patterns as separate entries
                myers_edit_distance(peq, text_window)
            } else {
                myers_edit_distance(peq, text_window)
            };
            if edit_dist > max_edits { continue; }

            // Traceback to classify
            // For reversed: we need to reverse the text window (semi-global free end)
            let tb_text = if pam_is_reversed {
                let mut rev = text_window.to_vec();
                rev.reverse();
                rev
            } else {
                text_window.to_vec()
            };
            let tb_pat = if pam_is_reversed {
                let mut rev = guide_patterns_ascii[p_idx].clone();
                rev.reverse();
                rev
            } else {
                guide_patterns_ascii[p_idx].clone()
            };
            let Some(align) = traceback(&tb_pat, &tb_text, max_edits) else { continue; };

            // Apply individual threshold checks
            if align.mismatches > max_mismatches { continue; }
            if align.dna_bulges > max_dna_bulges { continue; }
            if align.rna_bulges > max_rna_bulges { continue; }

            // Compute the chunk_idx: position of the guide match start in the chunk
            // For non-reversed: match ends at pam_start - 1, starts at pam_start - align.pattern_aligned length
            // Count non-gap chars in text_aligned = actual genome span
            let genome_span: usize = align.text_aligned.iter()
                .filter(|&&c| c != b'-').count();
            let match_chunk_idx = if pam_is_reversed {
                // For reversed, reverse mapping: genome match start is where aligned text reversed maps back
                // In tb_text (reversed), alignment starts at align.text_start. In original text, match ENDS at
                // text_window.len() - align.text_start and spans genome_span chars backward.
                let end_in_win = text_window.len() - align.text_start;
                win_start + end_in_win - genome_span
            } else {
                (pam_start - genome_span) as usize
            };

            matches.push(SearchMatch {
                chunk_idx: match_chunk_idx as u32,
                pattern_idx: p_idx as u32,
                mismatches: align.mismatches,
                dna_bulge_size: align.dna_bulges as u16,
                rna_bulge_size: align.rna_bulges as u16,
            });
        }
    }

    matches
}

fn pam_matches(genome: &[u8], filter: &[u8]) -> bool {
    if genome.len() != filter.len() { return false; }
    for (&g, &f) in genome.iter().zip(filter.iter()) {
        let gu = g.to_ascii_uppercase();
        let fu = f.to_ascii_uppercase();
        if fu == b'N' { continue; }
        // Mixed base check
        let g_bit = crate::bit4ops::STR_2_BIT4_public(gu);
        let f_bit = crate::bit4ops::STR_2_BIT4_public(fu);
        if (g_bit & f_bit) == 0 { return false; }
    }
    true
}
```

- [ ] **Step 2: Expose STR_2_BIT4 via a public helper**

In `cas-offinder-lib/src/bit4ops.rs`, add a public mapping function (the constant is private):

```rust
/// Map a single ASCII character to its 4-bit nucleotide encoding.
/// N/mixed bases expand to union of bits.
pub fn STR_2_BIT4_public(c: u8) -> u8 {
    STR_2_BIT4[true as usize][c as usize]
}
```

Actually rename to snake_case per Rust convention:

```rust
pub fn char_to_bit4(c: u8) -> u8 {
    STR_2_BIT4[true as usize][c as usize]
}
```

Update the call in `search.rs`:

```rust
        let g_bit = crate::bit4ops::char_to_bit4(gu);
        let f_bit = crate::bit4ops::char_to_bit4(fu);
```

- [ ] **Step 3: Build to catch errors**

Run: `cd cas-offinder-cli && RUSTFLAGS="-L$(pwd)" cargo build 2>&1 | tail -10`
Expected: compiles (may have warnings about unused `search_chunk_myers`)

- [ ] **Step 4: Commit**

```bash
git add cas-offinder-lib/src/search.rs cas-offinder-lib/src/bit4ops.rs
git commit -m "feat(search): add search_chunk_myers (Myers CPU path, unused)"
```

### Task 5.4: Wire search_chunk_myers into search() dispatch

**Files:**
- Modify: `cas-offinder-lib/src/search.rs`

- [ ] **Step 1: Update the `search()` public function signature to accept bulge params and search_filter**

Find `pub fn search(...)` in `src/search.rs`. Current signature:

```rust
pub fn search(
    devices: OclRunConfig,
    max_mismatches: u32,
    pattern_len: usize,
    patterns: &Vec<Vec<u8>>,
    recv: mpsc::Receiver<ChromChunkInfo>,
    dest: mpsc::SyncSender<Vec<Match>>,
)
```

New signature:

```rust
pub fn search(
    devices: OclRunConfig,
    max_mismatches: u32,
    max_dna_bulges: u32,
    max_rna_bulges: u32,
    search_filter: &[u8],
    pattern_len: usize,
    patterns: &Vec<Vec<u8>>,
    recv: mpsc::Receiver<ChromChunkInfo>,
    dest: mpsc::SyncSender<Vec<Match>>,
)
```

- [ ] **Step 2: Dispatch to search_chunk_myers on CPU path when any bulge > 0**

Inside `search()`, where the CPU dispatch happens (around line 495-502). Replace:

```rust
    if devices.is_empty() {
        search_compute_cpu(
            max_mismatches,
            pattern_len,
            patterns,
            compute_recv_src,
            compute_send_dest,
        );
    } else {
        match search_chunk_ocl(...)
```

With:

```rust
    use crate::bit4ops::{detect_pam_orientation, extract_guide};
    let (pam_is_reversed, pam_len) = detect_pam_orientation(search_filter);
    let guide_len = pattern_len - pam_len;

    // Extract guide ASCII from each pattern (patterns are currently ASCII vectors)
    let guide_patterns_ascii: Vec<Vec<u8>> = patterns.iter()
        .map(|p| extract_guide(p, pam_is_reversed, pam_len).to_vec())
        .collect();

    let search_filter_owned = search_filter.to_vec();

    if devices.is_empty() {
        search_compute_cpu_myers(
            max_mismatches,
            max_dna_bulges,
            max_rna_bulges,
            &search_filter_owned,
            pam_is_reversed,
            pam_len,
            guide_len,
            &guide_patterns_ascii,
            compute_recv_src,
            compute_send_dest,
        );
    } else {
        // GPU path: still use old algorithm for now (bulge params ignored).
        // Will be replaced in Phase 7.
        match search_chunk_ocl(
            devices,
            max_mismatches,
            pattern_len,
            patterns,
            compute_recv_src,
            compute_send_dest,
        ) {
            Ok(_) => {}
            Err(err_int) => { panic!("{}", err_int.to_string()) }
        };
    }
```

- [ ] **Step 3: Add search_compute_cpu_myers (similar to search_compute_cpu)**

Add in `src/search.rs`:

```rust
fn search_compute_cpu_myers(
    max_mismatches: u32,
    max_dna_bulges: u32,
    max_rna_bulges: u32,
    search_filter: &[u8],
    pam_is_reversed: bool,
    pam_len: usize,
    guide_len: usize,
    guide_patterns_ascii: &[Vec<u8>],
    recv: crossbeam_channel::Receiver<SearchChunkInfo>,
    dest: mpsc::SyncSender<SearchChunkResult>,
) {
    let patterns_arc = Arc::new(guide_patterns_ascii.to_vec());
    let filter_arc = Arc::new(search_filter.to_vec());
    let n_threads: usize = thread::available_parallelism().unwrap().into();
    let mut threads: Vec<thread::JoinHandle<()>> = Vec::new();
    for _ in 0..n_threads {
        let tpat = patterns_arc.clone();
        let tfilter = filter_arc.clone();
        let trecv = recv.clone();
        let tdest = dest.clone();
        threads.push(thread::spawn(move || {
            for schunk in trecv.iter() {
                let matches = search_chunk_myers(
                    max_mismatches,
                    max_dna_bulges,
                    max_rna_bulges,
                    &tfilter,
                    pam_is_reversed,
                    pam_len,
                    guide_len,
                    &tpat,
                    &schunk.data,
                );
                tdest.send(SearchChunkResult {
                    matches,
                    meta: schunk.meta,
                    data: schunk.data,
                }).unwrap();
            }
        }));
    }
    for t in threads { t.join().unwrap(); }
}
```

- [ ] **Step 4: Update main.rs to pass new args**

In `cas-offinder-cli/src/main.rs`, find the `search(` call and update:

```rust
    search(
        run_config,
        run_info.max_mismatches,
        run_info.max_dna_bulges,
        run_info.max_rna_bulges,
        &run_info.search_filter,
        run_info.pattern_len,
        &all_patterns_4bit,
        src_receiver,
        dest_sender,
    );
```

Wait — `all_patterns_4bit` is bit4-encoded patterns. But `search_chunk_myers` wants ASCII guide patterns. We need to convert.

Actually, the existing code passes `all_patterns_4bit` which is already bit4-encoded to GPU. For CPU Myers we want ASCII. Simplest fix: have search() accept ASCII patterns too, and convert to bit4 inside for GPU.

Actually looking at main.rs:

```rust
    let reversed_byte_patterns: Vec<Vec<u8>> = run_info.patterns.iter()
        .map(|v| reverse_compliment_char(v))
        .collect();
    let mut all_patterns: Vec<Vec<u8>> = run_info.patterns.clone();
    all_patterns.extend_from_slice(&reversed_byte_patterns);

    let all_patterns_4bit: Vec<Vec<u8>> = all_patterns.iter()
        .map(|pat| {
            let mut buf = vec![0_u8; cdiv(pat.len(), 2)];
            string_to_bit4(&mut buf, pat, 0, true);
            buf
        })
        .collect();

    search(run_config, run_info.max_mismatches, run_info.pattern_len,
           &all_patterns_4bit, src_receiver, dest_sender);
```

So `all_patterns` is ASCII, `all_patterns_4bit` is bit4-packed. Change `search()` signature to accept ASCII patterns. Inside search, convert to bit4 only when calling GPU.

Update `search()` signature:

```rust
pub fn search(
    devices: OclRunConfig,
    max_mismatches: u32,
    max_dna_bulges: u32,
    max_rna_bulges: u32,
    search_filter: &[u8],
    pattern_len: usize,
    patterns_ascii: &[Vec<u8>],  // ASCII patterns (not bit4)
    recv: mpsc::Receiver<ChromChunkInfo>,
    dest: mpsc::SyncSender<Vec<Match>>,
)
```

Inside `search()`, before the GPU path, convert to bit4:

```rust
    let patterns_bit4: Vec<Vec<u8>> = if !devices.is_empty() {
        patterns_ascii.iter().map(|pat| {
            let mut buf = vec![0_u8; cdiv(pat.len(), 2)];
            crate::bit4ops::string_to_bit4(&mut buf, pat, 0, true);
            buf
        }).collect()
    } else {
        Vec::new()
    };
```

And pass `&patterns_bit4` to `search_chunk_ocl` instead of `patterns` (adjust call).

Update main.rs:

```rust
    search(
        run_config,
        run_info.max_mismatches,
        run_info.max_dna_bulges,
        run_info.max_rna_bulges,
        &run_info.search_filter,
        run_info.pattern_len,
        &all_patterns,  // ASCII instead of all_patterns_4bit
        src_receiver,
        dest_sender,
    );
```

Remove the bit4 conversion from main.rs since `search()` handles it now.

- [ ] **Step 5: Build**

Run: `cd cas-offinder-cli && RUSTFLAGS="-L$(pwd)" cargo build --release 2>&1 | tail -10`
Expected: builds

- [ ] **Step 6: Regression test with bulge=0**

Run: `cd /bce/groups/pnucolab/analysis/cas-offinder-rust && LD_LIBRARY_PATH=./cas-offinder-cli:$LD_LIBRARY_PATH ./cas-offinder-cli/target/release/cas-offinder-cli test_run.in C /tmp/cpu_new.txt`
Compare: `diff <(sort /tmp/cpu_new.txt) <(sort output_cpu_test.txt)` (previous baseline)
Expected: no differences (117 matches, same content)

- [ ] **Step 7: Commit**

```bash
git add cas-offinder-lib/src/search.rs cas-offinder-cli/src/main.rs
git commit -m "feat(search): integrate Myers CPU path into search() dispatch"
```

### Task 5.5: Adjust SearchMatch bulge data flow to Match struct

**Files:**
- Modify: `cas-offinder-lib/src/search.rs`

- [ ] **Step 1: Update `convert_matches` to copy bulge sizes from SearchMatch to Match**

Find `fn convert_matches(...)` in `src/search.rs`. Update the Match construction:

```rust
            results.push(Match {
                chr_name: search_res.meta.chr_names[idx].clone(),
                chrom_idx: pos,
                pattern_idx: smatch.pattern_idx,
                mismatches: smatch.mismatches,
                is_forward: is_forward,
                dna_seq: dna_result,
                rna_seq: rna_result,
                dna_bulge_size: smatch.dna_bulge_size as u32,
                rna_bulge_size: smatch.rna_bulge_size as u32,
            });
```

- [ ] **Step 2: Build and run CPU bulge test**

First, create a bulge test input file:

```bash
cat > /tmp/bulge_test.in << 'EOF'
/bce/groups/pnucolab/analysis/cas-offinder-rust/cas-offinder-lib/tests/test_data/upstream1000.fa
NNNNNNNNNNNNNNNNNNNNNNGG 1 1
CCGTGGTTCAACATTTGCTTAGCA 5
EOF
```

Wait, this pattern has 24 chars but we said pattern_len must match filter_len. Make them consistent:

```bash
cat > /tmp/bulge_test.in << 'EOF'
/bce/groups/pnucolab/analysis/cas-offinder-rust/cas-offinder-lib/tests/test_data/upstream1000.fa
NNNNNNNNNNNNNNNNNNNNNNNNGG 1 1
CCGTGGTTCAACATTTGCTTAGCANNN 5
EOF
```

Actually the test_run.in has:
```
NNNNNNNNNNNNNNNNNNNNNNNN   (24 Ns, no PAM)
CCGTGGTTCAACATTTGCTTAGCA 11 (24nt, no PAM)
```

With `NNNNNNNNNNNNNNNNNNNNNNNN` (all N), the PAM orientation detection would see all Ns: left_ns=24, right_ns=24, so pam_len=0. That's valid.

For bulge test, add a realistic filter:

```bash
cat > /tmp/bulge_test.in << 'EOF'
/bce/groups/pnucolab/analysis/cas-offinder-rust/cas-offinder-lib/tests/test_data/upstream1000.fa
NNNNNNNNNNNNNNNNNNNNNNNNNGG 1 1
CCGTGGTTCAACATTTGCTTAGCANNN 5
EOF
```

Run: `LD_LIBRARY_PATH=./cas-offinder-cli:$LD_LIBRARY_PATH ./cas-offinder-cli/target/release/cas-offinder-cli /tmp/bulge_test.in C /tmp/bulge_out.txt`
Expected: produces output file with matches (some with bulge_size > 0)

- [ ] **Step 3: Inspect output**

Run: `wc -l /tmp/bulge_out.txt; head -20 /tmp/bulge_out.txt`
Expected: matches with various bulge configurations

- [ ] **Step 4: Commit**

```bash
git add cas-offinder-lib/src/search.rs
git commit -m "feat(search): flow bulge_size through convert_matches to Match"
```

---

## Phase 6: Output Format Update (cas-offinder-bulge Compatible)

Update main.rs to produce output in cas-offinder-bulge's format.

### Task 6.1: Rewrite output writer in main.rs

**Files:**
- Modify: `cas-offinder-cli/src/main.rs`

- [ ] **Step 1: Inspect current output logic**

Read lines 86-126 of `cas-offinder-cli/src/main.rs` to understand current output writing.

- [ ] **Step 2: Update the output writer to produce new format**

Replace the block inside `let result_count = thread::spawn(move || {...})` that writes output. The new format is:

```
#Bulge type\tcrRNA\tDNA\tChromosome\tPosition\tDirection\tMismatches\tBulge Size
```

One row per match.

New writer code:

```rust
    let result_count = thread::spawn(move || {
        let out_writer = if run_info.out_path != "-" {
            Box::new(File::create(&run_info.out_path).unwrap()) as Box<dyn Write>
        } else {
            Box::new(std::io::stdout()) as Box<dyn Write>
        };
        let mut out_buf_writer = BufWriter::new(out_writer);

        // Write header
        writeln!(
            out_buf_writer,
            "#Bulge type\tcrRNA\tDNA\tChromosome\tPosition\tDirection\tMismatches\tBulge Size"
        ).unwrap();

        let mut total_matches: u64 = 0;
        for chunk in dest_receiver.iter() {
            for m in chunk {
                let dir = if m.is_forward { '+' } else { '-' };
                let bulge_total = m.dna_bulge_size + m.rna_bulge_size;
                let bulge_type = if bulge_total == 0 {
                    "X"
                } else if m.dna_bulge_size > 0 {
                    "DNA"
                } else {
                    "RNA"
                };
                let rna_str = std::str::from_utf8(&m.rna_seq).unwrap();
                let dna_str = std::str::from_utf8(&m.dna_seq).unwrap();

                writeln!(
                    out_buf_writer,
                    "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
                    bulge_type, rna_str, dna_str, m.chr_name,
                    m.chrom_idx, dir, m.mismatches, bulge_total
                ).unwrap();
                total_matches += 1;
            }
        }
        total_matches
    });
```

Change the join to capture the count:

```rust
    let _total_matches = result_count.join().unwrap();
```

- [ ] **Step 3: Build**

Run: `cd cas-offinder-cli && RUSTFLAGS="-L$(pwd)" cargo build --release 2>&1 | tail -5`
Expected: builds

- [ ] **Step 4: Run on test data to see new output format**

Run: `cd /bce/groups/pnucolab/analysis/cas-offinder-rust && LD_LIBRARY_PATH=./cas-offinder-cli:$LD_LIBRARY_PATH ./cas-offinder-cli/target/release/cas-offinder-cli test_run.in C /tmp/output_new.txt`
Run: `head -5 /tmp/output_new.txt`
Expected: header + matches in new format

- [ ] **Step 5: Commit**

```bash
git add cas-offinder-cli/src/main.rs
git commit -m "feat(output): write cas-offinder-bulge compatible output format"
```

---

## Phase 7: Log File

Add `.log` file generation alongside output.

### Task 7.1: Add log_writer module

**Files:**
- Create: `cas-offinder-lib/src/log_writer.rs`
- Modify: `cas-offinder-lib/src/lib.rs`

- [ ] **Step 1: Create log_writer.rs**

```rust
// cas-offinder-lib/src/log_writer.rs
//! Log file generation for cas-offinder runs.

use std::fs::File;
use std::io::Write;
use std::path::Path;
use std::time::SystemTime;

pub struct RunLog {
    pub genome_path: String,
    pub genome_size: u64,
    pub n_patterns: usize,
    pub search_filter: String,
    pub max_mismatches: u32,
    pub max_dna_bulges: u32,
    pub max_rna_bulges: u32,
    pub device_label: String,
    pub pam_scan_elapsed: f64,
    pub myers_elapsed: f64,
    pub total_elapsed: f64,
    pub n_pam_sites: u64,
    pub n_matches: u64,
}

pub fn write_log<P: AsRef<Path>>(log_path: P, log: &RunLog) -> std::io::Result<()> {
    let mut f = File::create(log_path)?;
    let now = SystemTime::now();
    let secs = now.duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
    writeln!(f, "=== Cas-OFFinder Rust (Myers bit-parallel) ===")?;
    writeln!(f, "Run date (unix timestamp): {}", secs)?;
    writeln!(f)?;
    writeln!(f, "Device: {}", log.device_label)?;
    writeln!(f)?;
    writeln!(f, "Input:")?;
    writeln!(f, "  Genome: {}", log.genome_path)?;
    writeln!(f, "  Genome size: {} bp", log.genome_size)?;
    writeln!(f, "  Patterns: {}", log.n_patterns)?;
    writeln!(f, "  Search filter: {}", log.search_filter)?;
    writeln!(f, "  Max mismatches: {}", log.max_mismatches)?;
    writeln!(f, "  Max DNA bulges: {}", log.max_dna_bulges)?;
    writeln!(f, "  Max RNA bulges: {}", log.max_rna_bulges)?;
    writeln!(f)?;
    writeln!(f, "Phase 1 (PAM scan):")?;
    writeln!(f, "  NGG sites found: {}", log.n_pam_sites)?;
    writeln!(f, "  Elapsed: {:.3}s", log.pam_scan_elapsed)?;
    writeln!(f)?;
    writeln!(f, "Phase 2 (Myers edit distance):")?;
    writeln!(f, "  Candidates checked: {}", log.n_pam_sites)?;
    writeln!(f, "  Matches found: {}", log.n_matches)?;
    writeln!(f, "  Elapsed: {:.3}s", log.myers_elapsed)?;
    writeln!(f)?;
    writeln!(f, "Total elapsed: {:.3}s", log.total_elapsed)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_write_log() {
        let path = format!("/tmp/test_log_{}.log", std::process::id());
        let log = RunLog {
            genome_path: "/tmp/fake.fa".to_string(),
            genome_size: 1000,
            n_patterns: 2,
            search_filter: "NNNNNNN".to_string(),
            max_mismatches: 5,
            max_dna_bulges: 1,
            max_rna_bulges: 1,
            device_label: "CPU (4 threads)".to_string(),
            pam_scan_elapsed: 0.01,
            myers_elapsed: 0.1,
            total_elapsed: 0.12,
            n_pam_sites: 100,
            n_matches: 5,
        };
        write_log(&path, &log).unwrap();
        let content = std::fs::read_to_string(&path).unwrap();
        assert!(content.contains("=== Cas-OFFinder"));
        assert!(content.contains("Matches found: 5"));
        std::fs::remove_file(&path).ok();
    }
}
```

- [ ] **Step 2: Export from lib.rs**

Add:

```rust
mod log_writer;
pub use crate::log_writer::*;
```

- [ ] **Step 3: Run log_writer tests**

Run: `cd cas-offinder-lib && cargo test --lib log_writer::tests -- --nocapture`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add cas-offinder-lib/src/log_writer.rs cas-offinder-lib/src/lib.rs
git commit -m "feat(log): add RunLog struct and log file writer"
```

### Task 7.2: Track timing and stats in main.rs, write log file

**Files:**
- Modify: `cas-offinder-cli/src/main.rs`

- [ ] **Step 1: Track match count via the result_count join result**

Currently `result_count.join()` returns `()`. It already returns `u64` after Phase 6 changes. Confirm that.

- [ ] **Step 2: Track phase timing**

Currently only `start_time` and total elapsed are tracked. For phase-level timing, we'd need to instrument the search path. For the first version, approximate:
- `pam_scan_elapsed` = 0 (fused with Myers)
- `myers_elapsed` = total_elapsed minus output-writing overhead (approx = total)

Keep it simple for this first integration: set pam_scan_elapsed to 0 and myers_elapsed to total_elapsed.

- [ ] **Step 3: Write log at end of main**

Insert at the end of `main()` after `let tot_time = start_time.elapsed();`:

```rust
    let total_elapsed = tot_time.as_secs_f64();
    let genome_size = std::fs::metadata(&run_info.genome_path).map(|m| m.len()).unwrap_or(0);

    // Device label
    let device_label = if run_info.dev_ty_str == "G" {
        format!("GPU")
    } else if run_info.dev_ty_str == "C" {
        format!("CPU (native Rust)")
    } else {
        "Accelerator".to_string()
    };

    // ^ This requires storing dev_ty_str or device label. Easier approach: use dev_ty.
```

Actually, cleaner: capture device label from `OclRunConfig` earlier. Let me simplify:

```rust
    let log_path = if run_info.out_path != "-" {
        Some(format!("{}.log", &run_info.out_path))
    } else {
        None
    };

    if let Some(lpath) = log_path {
        let log = cas_offinder_lib::RunLog {
            genome_path: run_info.genome_path.clone(),
            genome_size: std::fs::metadata(&run_info.genome_path).map(|m| m.len()).unwrap_or(0),
            n_patterns: run_info.patterns.len(),
            search_filter: std::str::from_utf8(&run_info.search_filter).unwrap().to_string(),
            max_mismatches: run_info.max_mismatches,
            max_dna_bulges: run_info.max_dna_bulges,
            max_rna_bulges: run_info.max_rna_bulges,
            device_label: match run_info.dev_ty {
                OclDeviceType::CPU => "CPU".to_string(),
                OclDeviceType::GPU => "GPU".to_string(),
                OclDeviceType::ACCEL => "Accelerator".to_string(),
                OclDeviceType::ALL => "All".to_string(),
            },
            pam_scan_elapsed: 0.0,
            myers_elapsed: total_elapsed,
            total_elapsed,
            n_pam_sites: 0, // not tracked yet
            n_matches: _total_matches,
        };
        let _ = cas_offinder_lib::write_log(&lpath, &log);
    }

    eprintln!("Completed in {}s", total_elapsed);
```

- [ ] **Step 4: Build and test**

Run: `cd cas-offinder-cli && RUSTFLAGS="-L$(pwd)" cargo build --release 2>&1 | tail -5`
Run: `cd /bce/groups/pnucolab/analysis/cas-offinder-rust && LD_LIBRARY_PATH=./cas-offinder-cli:$LD_LIBRARY_PATH ./cas-offinder-cli/target/release/cas-offinder-cli test_run.in C /tmp/new_out.txt`
Run: `cat /tmp/new_out.txt.log`
Expected: log file is created and contains device info, stats, timing.

- [ ] **Step 5: Commit**

```bash
git add cas-offinder-cli/src/main.rs
git commit -m "feat(log): write .log file alongside output"
```

---

## Phase 8: GPU Kernel with Myers + PAM

Replace the OpenCL kernel with a Myers-based kernel. This phase makes GPU output match CPU output with bulge support.

### Task 8.1: Write OpenCL Myers kernel stub

**Files:**
- Create: `cas-offinder-lib/src/kernel_myers.cl`
- Modify: `cas-offinder-lib/src/search.rs`

- [ ] **Step 1: Create kernel_myers.cl**

```c
// cas-offinder-lib/src/kernel_myers.cl
// Myers bit-parallel edit distance kernel with PAM exact-match pre-filter.

#ifndef GUIDE_LEN
#error "GUIDE_LEN must be defined at compile time"
#endif

#ifndef PAM_LEN
#error "PAM_LEN must be defined at compile time"
#endif

#ifndef MAX_EDITS
#error "MAX_EDITS must be defined at compile time"
#endif

// Nucleotide to index (A=0, C=1, G=2, T=3). Returns 4 for N.
// Input is bit-4 encoded char: T=0x1, C=0x2, A=0x4, G=0x8, N=0xF
int nucl_idx_bit4(uchar b4) {
    // If multiple bits (N or mixed), return 4 (match all)
    int pop = popcount((uint)b4);
    if (pop != 1) return 4;
    if (b4 == 0x4) return 0; // A
    if (b4 == 0x2) return 1; // C
    if (b4 == 0x8) return 2; // G
    if (b4 == 0x1) return 3; // T
    return 4;
}

// Get nucleotide bit4 from packed chunk at position i
uchar get_bit4(__global const uchar* chunk, uint i) {
    uchar byte = chunk[i / 2];
    if (i % 2 == 0) return byte & 0x0F;
    else return (byte >> 4) & 0x0F;
}

struct s_match {
    uint loc;
    uint pattern_idx;
    uint mismatches;
    ushort dna_bulge_size;
    ushort rna_bulge_size;
};

__kernel void find_matches_myers(
    __global const uchar* genome_bit4,       // bit4-packed genome chunk
    __global const ulong* peq_tables,        // 4 * n_patterns u64 (Peq per base per pattern)
    __global const uchar* pam_filter_bit4,   // PAM filter in bit4, length = PAM_LEN
    uint max_mismatches,
    uint max_dna_bulges,
    uint max_rna_bulges,
    uint pam_is_reversed,
    uint total_nucl,
    __global struct s_match* out_matches,
    __global uint* out_count
) {
    uint j = get_global_id(0);  // genome position (guide start)
    uint p = get_global_id(1);  // pattern index

    // Bounds check
    if (j + GUIDE_LEN + PAM_LEN > total_nucl) return;

    // Phase 1: PAM check
    uint pam_start = pam_is_reversed ? j : (j + GUIDE_LEN);
    for (uint k = 0; k < PAM_LEN; k++) {
        uchar g = get_bit4(genome_bit4, pam_start + k);
        uchar f = pam_filter_bit4[k];
        if ((g & f) == 0) return;
    }

    // Phase 2: Myers
    ulong peq[4];
    peq[0] = peq_tables[p * 4 + 0];
    peq[1] = peq_tables[p * 4 + 1];
    peq[2] = peq_tables[p * 4 + 2];
    peq[3] = peq_tables[p * 4 + 3];

    ulong vp = (GUIDE_LEN < 64) ? (((ulong)1 << GUIDE_LEN) - 1) : ~(ulong)0;
    ulong vn = 0;
    int score = GUIDE_LEN;
    ulong last_bit = (ulong)1 << (GUIDE_LEN - 1);

    // Text window: guide_len + max_dna_bulges characters
    uint myers_text_len = GUIDE_LEN + max_dna_bulges;
    uint win_start = pam_is_reversed ? (pam_start + PAM_LEN) : (pam_start >= myers_text_len ? pam_start - myers_text_len : 0);
    uint win_end = pam_is_reversed ? min(win_start + myers_text_len, total_nucl) : pam_start;

    // For reversed: iterate text window left to right but we want alignment ending at win_end
    // For non-reversed: iterate text window left to right, alignment ends at win_end
    // In both cases Myers semi-global "ending at text[win_end]" is what we compute.

    for (uint t = win_start; t < win_end; t++) {
        uchar g_bit4 = get_bit4(genome_bit4, t);
        ulong eq = 0;
        if (g_bit4 & 0x4) eq |= peq[0];
        if (g_bit4 & 0x2) eq |= peq[1];
        if (g_bit4 & 0x8) eq |= peq[2];
        if (g_bit4 & 0x1) eq |= peq[3];
        ulong x = eq | vn;
        ulong d0 = (((x & vp) + vp) ^ vp) | x;
        ulong hn = vp & d0;
        ulong hp = vn | ~(vp | d0);
        ulong x_shift = (hp << 1) | 1;
        vn = x_shift & d0;
        vp = (hn << 1) | ~(x_shift | d0);
        if (hp & last_bit) score++;
        if (hn & last_bit) score--;
    }

    if (score < 0 || (uint)score > MAX_EDITS) return;

    // Record candidate; traceback happens on CPU.
    // Approximate bulge sizes as 0 here — CPU will reclassify.
    uint idx = atomic_inc(out_count);
    if (idx < (1u << 22)) {
        out_matches[idx].loc = j;
        out_matches[idx].pattern_idx = p;
        out_matches[idx].mismatches = (uint)score;
        out_matches[idx].dna_bulge_size = 0;
        out_matches[idx].rna_bulge_size = 0;
    }
}
```

- [ ] **Step 2: Embed the kernel source via include_str!**

In `src/search.rs`, find where `KERNEL_CONTENTS` is defined (usually `include_str!("kernel.cl")`). Add a new constant:

```rust
const KERNEL_MYERS_CONTENTS: &str = include_str!("kernel_myers.cl");
```

- [ ] **Step 3: Build (sanity check only — not used yet)**

Run: `cd cas-offinder-cli && RUSTFLAGS="-L$(pwd)" cargo build --release 2>&1 | tail -5`
Expected: builds (kernel_myers.cl embedded as string)

- [ ] **Step 4: Commit**

```bash
git add cas-offinder-lib/src/kernel_myers.cl cas-offinder-lib/src/search.rs
git commit -m "feat(gpu): add Myers OpenCL kernel source (not wired yet)"
```

### Task 8.2: Build Peq tables for GPU and write search_chunk_ocl_myers

**Files:**
- Modify: `cas-offinder-lib/src/search.rs`

- [ ] **Step 1: Add helper to build Peq table array for GPU**

```rust
fn build_peq_array(guide_patterns_ascii: &[Vec<u8>]) -> Vec<u64> {
    let mut out = Vec::with_capacity(guide_patterns_ascii.len() * 4);
    for pat in guide_patterns_ascii {
        let peq = crate::myers::build_peq(pat);
        out.push(peq.peq[0]);
        out.push(peq.peq[1]);
        out.push(peq.peq[2]);
        out.push(peq.peq[3]);
    }
    out
}
```

- [ ] **Step 2: Write new GPU search function (`search_device_ocl_myers`)**

Similar structure to `search_device_ocl`, but uses new kernel:

```rust
unsafe fn search_device_ocl_myers(
    max_mismatches: u32,
    max_dna_bulges: u32,
    max_rna_bulges: u32,
    search_filter_bit4: Arc<Vec<u8>>,
    pam_is_reversed: bool,
    pam_len: usize,
    guide_len: usize,
    peq_array: Arc<Vec<u64>>,
    n_patterns: usize,
    context: Arc<context::Context>,
    program: Arc<program::Program>,
    dev: Arc<device::Device>,
    recv: crossbeam_channel::Receiver<SearchChunkInfo>,
    dest: mpsc::SyncSender<SearchChunkResult>,
) -> Result<()> {
    const OUT_BUF_SIZE: usize = 1 << 22;
    const CL_BLOCK: u32 = 1;
    const CL_NO_BLOCK: u32 = 0;
    let queue = command_queue::CommandQueue::create(&context, dev.id(), 0)?;
    let kernel = kernel::Kernel::create(&program, "find_matches_myers")?;
    let mut genome_bufs = create_ocl_bufs::<u8>(&context, SEARCH_CHUNK_SIZE_BYTES)?;
    let mut out_counts = create_ocl_bufs::<u32>(&context, 1)?;
    let mut out_bufs = create_ocl_bufs::<SearchMatch>(&context, OUT_BUF_SIZE)?;
    let mut peq_buf = create_ocl_buf::<u64>(&context, peq_array.len())?;
    queue.enqueue_write_buffer(&mut peq_buf, CL_BLOCK, 0, &peq_array, &[])?;
    let mut pam_filter_buf = create_ocl_buf::<u8>(&context, pam_len)?;
    queue.enqueue_write_buffer(&mut pam_filter_buf, CL_BLOCK, 0, &search_filter_bit4, &[])?;

    let max_edits = max_mismatches + max_dna_bulges + max_rna_bulges;
    let total_nucl: u32 = (SEARCH_CHUNK_SIZE_BYTES * 2) as u32;

    for item in recv.iter() {
        let n_chunks = std::cmp::min(CHUNKS_PER_SEARCH - 1, item.meta.chr_names.len());
        let n_genome_bytes = n_chunks * CHUNK_SIZE_BYTES;
        let cur_genome_buf = &mut genome_bufs[0];
        let cur_size_buf = &mut out_counts[0];
        let cur_out_buf = &mut out_bufs[0];
        let write_event = queue.enqueue_write_buffer(
            cur_genome_buf, CL_NO_BLOCK, 0,
            &item.data[..n_genome_bytes + CHUNK_SIZE_BYTES], &[])?;
        let clear_count_event = queue.enqueue_write_buffer(
            cur_size_buf, CL_NO_BLOCK, 0, &[0], &[])?;

        let kernel_event = kernel::ExecuteKernel::new(&kernel)
            .set_arg(cur_genome_buf)
            .set_arg(&peq_buf)
            .set_arg(&pam_filter_buf)
            .set_arg(&max_mismatches)
            .set_arg(&max_dna_bulges)
            .set_arg(&max_rna_bulges)
            .set_arg(&(pam_is_reversed as u32))
            .set_arg(&total_nucl)
            .set_arg(cur_out_buf)
            .set_arg(cur_size_buf)
            .set_global_work_sizes(&[total_nucl as usize, n_patterns])
            .set_wait_event(&write_event)
            .set_wait_event(&clear_count_event)
            .enqueue_nd_range(&queue)?;
        let mut readsize_buf = [0u32];
        queue.enqueue_read_buffer(cur_size_buf, CL_BLOCK, 0, &mut readsize_buf, &[kernel_event.get()])?;
        let readsize = readsize_buf[0];
        if readsize != 0 {
            let mut outvec: Vec<SearchMatch> = vec![SearchMatch {
                chunk_idx: 0, pattern_idx: 0, mismatches: 0,
                dna_bulge_size: 0, rna_bulge_size: 0,
            }; readsize as usize];
            queue.enqueue_read_buffer(cur_out_buf, CL_BLOCK, 0, &mut outvec[..], &[])?;

            // CPU-side traceback pass to classify mismatches vs bulges
            // (each GPU-found candidate → run traceback on ASCII view)
            // For speed we keep genome as bit4; convert a small window on the fly
            // ... actually this is complex. For now, defer traceback to post-processing.

            dest.send(SearchChunkResult {
                matches: outvec, meta: item.meta, data: item.data,
            }).unwrap();
        }
    }
    Ok(())
}
```

- [ ] **Step 3: Dispatch in `search()` when devices are available**

In the GPU path of `search()`, replace the existing `search_chunk_ocl(...)` call with `search_chunk_ocl_myers(...)` analogous to task 5.4. The function should:
- Build Peq array from ASCII guide patterns
- Pre-encode search filter to bit4
- Compile program with `-DGUIDE_LEN=... -DPAM_LEN=... -DMAX_EDITS=...`
- Spawn one `search_device_ocl_myers` per device

Write the wrapper `search_chunk_ocl_myers(...)`:

```rust
fn search_chunk_ocl_myers(
    devices: OclRunConfig,
    max_mismatches: u32,
    max_dna_bulges: u32,
    max_rna_bulges: u32,
    search_filter: &[u8],
    pam_is_reversed: bool,
    pam_len: usize,
    guide_len: usize,
    guide_patterns_ascii: &[Vec<u8>],
    recv: crossbeam_channel::Receiver<SearchChunkInfo>,
    dest: mpsc::SyncSender<SearchChunkResult>,
) -> Result<()> {
    let peq_arc = Arc::new(build_peq_array(guide_patterns_ascii));
    let n_patterns = guide_patterns_ascii.len();

    let mut filter_bit4 = vec![0u8; pam_len];
    let pam_ascii: Vec<u8> = if pam_is_reversed {
        search_filter[..pam_len].to_vec()
    } else {
        search_filter[search_filter.len() - pam_len..].to_vec()
    };
    for (i, &c) in pam_ascii.iter().enumerate() {
        filter_bit4[i] = crate::bit4ops::char_to_bit4(c);
    }
    let filter_arc = Arc::new(filter_bit4);

    let max_edits = max_mismatches + max_dna_bulges + max_rna_bulges;
    let compile_defs = format!(
        " -DGUIDE_LEN={} -DPAM_LEN={} -DMAX_EDITS={}",
        guide_len, pam_len, max_edits
    );

    let mut threads: Vec<JoinHandle<Result<()>>> = Vec::new();
    for (_, devs) in devices.get().iter() {
        let plat_devs: Vec<*mut std::ffi::c_void> = devs.iter().map(|d| d.id()).collect();
        if !plat_devs.is_empty() {
            let context = Arc::new(unsafe {
                context::Context::from_devices(&plat_devs, &[0], None, null_mut())?
            });
            let p_devices: Vec<Arc<device::Device>> = plat_devs.iter()
                .map(|d| Arc::new(device::Device::new(*d))).collect();
            let program = Arc::new(
                program::Program::create_and_build_from_source(
                    &context, KERNEL_MYERS_CONTENTS, &compile_defs
                ).map_err(|err| { eprintln!("{}", err); }).unwrap()
            );

            for p_dev in p_devices {
                let t_dest = dest.clone();
                let t_recv = recv.clone();
                let t_context = context.clone();
                let t_prog = program.clone();
                let t_peq = peq_arc.clone();
                let t_filter = filter_arc.clone();
                threads.push(thread::spawn(move || {
                    unsafe {
                        search_device_ocl_myers(
                            max_mismatches, max_dna_bulges, max_rna_bulges,
                            t_filter, pam_is_reversed, pam_len, guide_len,
                            t_peq, n_patterns,
                            t_context, t_prog, p_dev,
                            t_recv, t_dest
                        )
                    }
                }));
            }
        }
    }
    for t in threads { t.join().unwrap()?; }
    Ok(())
}
```

- [ ] **Step 4: Wire into `search()` dispatch**

In `search()`, replace the existing GPU branch:

```rust
    } else {
        // GPU path: still use old algorithm for now
        match search_chunk_ocl(...)
    }
```

With:

```rust
    } else {
        match search_chunk_ocl_myers(
            devices,
            max_mismatches,
            max_dna_bulges,
            max_rna_bulges,
            &search_filter_owned,
            pam_is_reversed,
            pam_len,
            guide_len,
            &guide_patterns_ascii,
            compute_recv_src,
            compute_send_dest,
        ) {
            Ok(_) => {}
            Err(err_int) => panic!("{}", err_int.to_string()),
        };
    }
```

- [ ] **Step 5: Build**

Run: `cd cas-offinder-cli && RUSTFLAGS="-L$(pwd)" cargo build --release 2>&1 | tail -15`
Expected: builds (may have warnings)

- [ ] **Step 6: Run GPU test**

Run: `cd /bce/groups/pnucolab/analysis/cas-offinder-rust && LD_LIBRARY_PATH=./cas-offinder-cli:$LD_LIBRARY_PATH ./cas-offinder-cli/target/release/cas-offinder-cli test_run.in G /tmp/gpu_out.txt`
Expected: produces output

- [ ] **Step 7: Compare GPU output to CPU output**

Run: `LD_LIBRARY_PATH=./cas-offinder-cli:$LD_LIBRARY_PATH ./cas-offinder-cli/target/release/cas-offinder-cli test_run.in C /tmp/cpu_out.txt`
Run: `diff <(sort /tmp/gpu_out.txt) <(sort /tmp/cpu_out.txt)`
Expected: no differences (or only minor output ordering)

- [ ] **Step 8: Commit**

```bash
git add cas-offinder-lib/src/search.rs cas-offinder-lib/src/kernel_myers.cl
git commit -m "feat(gpu): wire Myers kernel into GPU search path"
```

### Task 8.3: Add CPU traceback pass for GPU-found matches

**Files:**
- Modify: `cas-offinder-lib/src/search.rs`

- [ ] **Step 1: After GPU returns raw matches, run traceback to classify**

GPU's `search_device_ocl_myers` currently sets bulge sizes to 0. We need to run traceback on each match (CPU-side) to classify.

Add a helper that takes `SearchChunkResult` from GPU and re-runs traceback:

```rust
fn reclassify_gpu_matches(
    results: &mut SearchChunkResult,
    guide_patterns_ascii: &[Vec<u8>],
    pam_is_reversed: bool,
    pam_len: usize,
    guide_len: usize,
    max_mismatches: u32,
    max_dna_bulges: u32,
    max_rna_bulges: u32,
) {
    let max_edits = max_mismatches + max_dna_bulges + max_rna_bulges;
    let myers_text_len = guide_len + max_dna_bulges as usize;

    // Decode genome once for the chunk
    let mut ascii_chunk = vec![0u8; SEARCH_CHUNK_SIZE_BYTES * 2];
    crate::bit4ops::bit4_to_string(&mut ascii_chunk, &results.data[..], 0, ascii_chunk.len());

    let mut filtered = Vec::new();
    for smatch in results.matches.drain(..) {
        let j = smatch.chunk_idx as usize;
        let pam_start = if pam_is_reversed { j } else { j + guide_len };
        let (win_start, win_end) = if pam_is_reversed {
            let s = pam_start + pam_len;
            let e = (s + myers_text_len).min(ascii_chunk.len());
            (s, e)
        } else {
            let e = pam_start;
            let s = e.saturating_sub(myers_text_len);
            (s, e)
        };
        let text_window = &ascii_chunk[win_start..win_end];
        let (pat, txt): (Vec<u8>, Vec<u8>) = if pam_is_reversed {
            let mut rp = guide_patterns_ascii[smatch.pattern_idx as usize].clone();
            let mut rt = text_window.to_vec();
            rp.reverse();
            rt.reverse();
            (rp, rt)
        } else {
            (guide_patterns_ascii[smatch.pattern_idx as usize].clone(), text_window.to_vec())
        };
        let Some(align) = crate::traceback::traceback(&pat, &txt, max_edits) else { continue; };
        if align.mismatches > max_mismatches { continue; }
        if align.dna_bulges > max_dna_bulges { continue; }
        if align.rna_bulges > max_rna_bulges { continue; }

        let genome_span: usize = align.text_aligned.iter().filter(|&&c| c != b'-').count();
        let final_chunk_idx = if pam_is_reversed {
            let end_in_win = text_window.len() - align.text_start;
            win_start + end_in_win - genome_span
        } else {
            pam_start - genome_span
        };

        filtered.push(SearchMatch {
            chunk_idx: final_chunk_idx as u32,
            pattern_idx: smatch.pattern_idx,
            mismatches: align.mismatches,
            dna_bulge_size: align.dna_bulges as u16,
            rna_bulge_size: align.rna_bulges as u16,
        });
    }
    results.matches = filtered;
}
```

Wire the reclassification into the GPU path: apply `reclassify_gpu_matches` after `search_device_ocl_myers` returns a `SearchChunkResult` but before forwarding to `dest`. Modify `search_device_ocl_myers` so it accepts the guide patterns and calls this helper before `dest.send(...)`.

The cleanest structural change: make `search_device_ocl_myers` take `guide_patterns: Arc<Vec<Vec<u8>>>` and inline-call `reclassify_gpu_matches(&mut result, ...)` before `dest.send(result)`. Update the `search_chunk_ocl_myers` wrapper accordingly.

- [ ] **Step 2: Build & test**

Run: `cd cas-offinder-cli && RUSTFLAGS="-L$(pwd)" cargo build --release 2>&1 | tail -5`
Run: GPU vs CPU comparison again
Expected: outputs match

- [ ] **Step 3: Commit**

```bash
git add cas-offinder-lib/src/search.rs
git commit -m "feat(gpu): reclassify GPU matches via CPU traceback for bulge classification"
```

---

## Phase 9: Integration Tests

### Task 9.1: Add bulge-aware integration test

**Files:**
- Create: `cas-offinder-lib/tests/integration_bulge.rs`

- [ ] **Step 1: Write integration test harness**

This test compares our output to an expected set of matches. Because we don't have easy access to cas-offinder-bulge to generate ground truth here, we'll:
1. Run the binary on known small input
2. Verify match count and format

```rust
// cas-offinder-lib/tests/integration_bulge.rs
use std::process::Command;
use std::path::Path;

fn binary_path() -> String {
    // Adjust path as needed
    format!("{}/../cas-offinder-cli/target/release/cas-offinder-cli",
        env!("CARGO_MANIFEST_DIR"))
}

#[test]
#[ignore] // Requires built release binary; run manually
fn integration_bulge_small_file() {
    let input = format!("{}/tests/test_data/bulge_small.in", env!("CARGO_MANIFEST_DIR"));
    let output = "/tmp/rust_bulge_out.txt";

    let status = Command::new(binary_path())
        .arg(&input)
        .arg("C")
        .arg(output)
        .env("LD_LIBRARY_PATH",
             format!("{}/../cas-offinder-cli", env!("CARGO_MANIFEST_DIR")))
        .status()
        .expect("Failed to run binary");
    assert!(status.success());

    let content = std::fs::read_to_string(output).unwrap();
    let lines: Vec<&str> = content.lines().collect();
    assert!(lines[0].starts_with("#Bulge type"));
    assert!(lines.len() > 1, "Expected at least one match");

    // Verify at least one "DNA" or "RNA" bulge type is present
    let has_bulge = lines[1..].iter().any(|l| l.starts_with("DNA") || l.starts_with("RNA"));
    // Not strictly required, but likely with the test data
    eprintln!("has_bulge: {}", has_bulge);
}
```

- [ ] **Step 2: Create test input file**

```bash
mkdir -p cas-offinder-lib/tests/test_data
cat > cas-offinder-lib/tests/test_data/bulge_small.in << 'EOF'
/bce/groups/pnucolab/analysis/cas-offinder-rust/cas-offinder-lib/tests/test_data/upstream1000.fa
NNNNNNNNNNNNNNNNNNNNNNNNNGG 1 1
CCGTGGTTCAACATTTGCTTAGCANNN 5
EOF
```

- [ ] **Step 3: Run the integration test**

Run: `cd cas-offinder-lib && cargo test --test integration_bulge -- --ignored --nocapture`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add cas-offinder-lib/tests/integration_bulge.rs cas-offinder-lib/tests/test_data/bulge_small.in
git commit -m "test(integration): add bulge integration test"
```

---

## Phase 10: Benchmarks

### Task 10.1: Run before/after benchmarks

**Files:**
- Create: `docs/superpowers/benchmarks-2026-04-16.md`

- [ ] **Step 1: Record baseline (already done earlier)**

Baseline on upstream1000.fa:
- GPU: ~0.49s
- CPU: ~0.20s

- [ ] **Step 2: Run new implementation with bulge=0 (regression check)**

Run three times each:
```bash
cd /bce/groups/pnucolab/analysis/cas-offinder-rust
for i in 1 2 3; do LD_LIBRARY_PATH=./cas-offinder-cli:$LD_LIBRARY_PATH ./cas-offinder-cli/target/release/cas-offinder-cli test_run.in G /tmp/g.txt; done
for i in 1 2 3; do LD_LIBRARY_PATH=./cas-offinder-cli:$LD_LIBRARY_PATH ./cas-offinder-cli/target/release/cas-offinder-cli test_run.in C /tmp/c.txt; done
```

- [ ] **Step 3: Run new implementation with bulge=1,1**

Use test input:
```bash
cat > /tmp/bench_bulge.in << 'EOF'
/bce/groups/pnucolab/analysis/cas-offinder-rust/cas-offinder-lib/tests/test_data/upstream1000.fa
NNNNNNNNNNNNNNNNNNNNNNNNNGG 1 1
CCGTGGTTCAACATTTGCTTAGCANNN 5
EOF

for i in 1 2 3; do LD_LIBRARY_PATH=./cas-offinder-cli:$LD_LIBRARY_PATH ./cas-offinder-cli/target/release/cas-offinder-cli /tmp/bench_bulge.in G /tmp/g_bulge.txt; done
for i in 1 2 3; do LD_LIBRARY_PATH=./cas-offinder-cli:$LD_LIBRARY_PATH ./cas-offinder-cli/target/release/cas-offinder-cli /tmp/bench_bulge.in C /tmp/c_bulge.txt; done
```

- [ ] **Step 4: Write benchmark report**

Create `docs/superpowers/benchmarks-2026-04-16.md` with the results in a table:

```markdown
# Benchmark: Myers Bit-Parallel Bulge Support

Test data: `cas-offinder-lib/tests/test_data/upstream1000.fa` (~13 KB)

## Results

| Configuration | GPU time (avg) | CPU time (avg) |
|---|---|---|
| Baseline (popcount, no bulge) | 0.49s | 0.20s |
| Myers (bulge=0) | ? | ? |
| Myers (bulge=1,1) | ? | ? |

## Notes

- First-run GPU has kernel compilation overhead (~0.3s)
- CPU uses native Rust multi-threaded implementation
- [Describe any performance observations]
```

Fill in measured values.

- [ ] **Step 5: Commit**

```bash
git add docs/superpowers/benchmarks-2026-04-16.md
git commit -m "docs: benchmark results for Myers bulge support"
```

---

## Self-Review Notes

After finishing all tasks, run the full test suite:

```bash
cd cas-offinder-cli && RUSTFLAGS="-L$(pwd)" cargo test 2>&1 | tail -20
```

All tests should pass. If any fail, fix before claiming completion.

**Known simplifications in this plan:**
- Phase 1 GPU kernel calls CPU traceback for bulge classification. Future work could move traceback to GPU.
- Timing only tracks total elapsed, not phase-wise. Adding phase timers is a small follow-up.
- Pattern length limited to 64nt (single u64 for Peq bit-vector). Multi-word Myers is a future extension.

**Unresolved items to watch during implementation:**
- If genome decoding from bit4 to ASCII per chunk is too slow, specialize Myers to consume bit4 directly (see bit4ops module).
- OUT_BUF_SIZE (1<<22) may need to grow for high-bulge runs on large genomes.
