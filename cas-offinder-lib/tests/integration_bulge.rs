//! End-to-end integration tests for cas-offinder-rust's bulge support.
//!
//! These tests drive the release-built CLI binary on the shipping
//! `upstream1000.fa` test genome and assert on the produced output file.
//! They are marked `#[ignore]` because they depend on a pre-built release
//! binary (`cas-offinder-cli/target/release/cas-offinder-cli`) and the
//! `LD_LIBRARY_PATH` pointing at `cas-offinder-cli/` so `libOpenCL.so`
//! resolves.
//!
//! To run:
//! ```text
//! cd cas-offinder-cli && RUSTFLAGS="-L$(pwd)" cargo build --release
//! cd ../cas-offinder-lib && cargo test --test integration_bulge -- --ignored --nocapture
//! ```

use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Command;

fn repo_root() -> PathBuf {
    // Manifest dir is cas-offinder-lib/. The repo root is its parent.
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    manifest.parent().unwrap().to_path_buf()
}

fn binary_path() -> PathBuf {
    repo_root().join("cas-offinder-cli/target/release/cas-offinder-cli")
}

fn lib_path() -> PathBuf {
    repo_root().join("cas-offinder-cli")
}

fn genome_path() -> PathBuf {
    repo_root().join("cas-offinder-lib/tests/test_data/upstream1000.fa")
}

fn make_temp_input(contents: &str, tag: &str) -> PathBuf {
    let path = std::env::temp_dir().join(format!(
        "cof_integ_{}_{}_{}.in",
        tag,
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    let mut f = fs::File::create(&path).unwrap();
    f.write_all(contents.as_bytes()).unwrap();
    path
}

fn run_cli(input: &PathBuf, device: &str, output: &PathBuf) -> String {
    let status = Command::new(binary_path())
        .arg(input)
        .arg(device)
        .arg(output)
        .env("LD_LIBRARY_PATH", lib_path())
        .status()
        .expect("failed to execute cas-offinder-cli (build release first)");
    assert!(status.success(), "binary exited non-zero on {:?}", input);
    fs::read_to_string(output).unwrap()
}

fn count_matches(output: &str) -> usize {
    // First line is header starting with "#"; data rows follow.
    output.lines().filter(|l| !l.starts_with('#') && !l.is_empty()).count()
}

fn has_header(output: &str) -> bool {
    output
        .lines()
        .next()
        .map(|l| l.starts_with("#Bulge type"))
        .unwrap_or(false)
}

// ---------------------------------------------------------------------------
// Legacy popcount path (bulge=0): must match baseline of 117 matches
// ---------------------------------------------------------------------------

#[test]
#[ignore]
fn integration_bulge_zero_cpu_baseline() {
    let input_contents = format!(
        "{}\nNNNNNNNNNNNNNNNNNNNNNNNN\nCCGTGGTTCAACATTTGCTTAGCA 11\nGATGTTGGTAAGTGGGATATGGCA 11\n",
        genome_path().display()
    );
    let in_path = make_temp_input(&input_contents, "baseline_cpu");
    let out_path = in_path.with_extension("out");
    let content = run_cli(&in_path, "C", &out_path);
    assert!(has_header(&content), "header missing");
    assert_eq!(
        count_matches(&content),
        117,
        "baseline should produce 117 matches"
    );
    // Cleanup
    fs::remove_file(&in_path).ok();
    fs::remove_file(&out_path).ok();
    fs::remove_file(format!("{}.log", out_path.display())).ok();
}

#[test]
#[ignore]
fn integration_bulge_zero_gpu_matches_cpu() {
    let input_contents = format!(
        "{}\nNNNNNNNNNNNNNNNNNNNNNNNN\nCCGTGGTTCAACATTTGCTTAGCA 11\nGATGTTGGTAAGTGGGATATGGCA 11\n",
        genome_path().display()
    );
    let in_path = make_temp_input(&input_contents, "zero_crossdev");
    let cpu_out = in_path.with_extension("cpu.out");
    let gpu_out = in_path.with_extension("gpu.out");
    let cpu_content = run_cli(&in_path, "C", &cpu_out);
    let gpu_content = run_cli(&in_path, "G", &gpu_out);

    let mut cpu_rows: Vec<&str> = cpu_content.lines().filter(|l| !l.starts_with('#')).collect();
    let mut gpu_rows: Vec<&str> = gpu_content.lines().filter(|l| !l.starts_with('#')).collect();
    cpu_rows.sort();
    gpu_rows.sort();
    assert_eq!(cpu_rows, gpu_rows, "CPU and GPU must produce identical matches");

    fs::remove_file(&in_path).ok();
    fs::remove_file(&cpu_out).ok();
    fs::remove_file(&gpu_out).ok();
    fs::remove_file(format!("{}.log", cpu_out.display())).ok();
    fs::remove_file(format!("{}.log", gpu_out.display())).ok();
}

// ---------------------------------------------------------------------------
// Myers path (bulge > 0): CPU vs GPU must agree; output format is correct
// ---------------------------------------------------------------------------

#[test]
#[ignore]
fn integration_bulge_myers_cpu_gpu_match() {
    // NGG PAM, 1 DNA bulge + 1 RNA bulge, max 5 mismatches.
    let input_contents = format!(
        "{}\nNNNNNNNNNNNNNNNNNNNNNNNNNGG 1 1\nCCGTGGTTCAACATTTGCTTAGCANNN 5\n",
        genome_path().display()
    );
    let in_path = make_temp_input(&input_contents, "bulge_crossdev");
    let cpu_out = in_path.with_extension("cpu.out");
    let gpu_out = in_path.with_extension("gpu.out");
    let cpu_content = run_cli(&in_path, "C", &cpu_out);
    let gpu_content = run_cli(&in_path, "G", &gpu_out);

    assert!(has_header(&cpu_content));
    assert!(has_header(&gpu_content));
    let mut cpu_rows: Vec<&str> = cpu_content.lines().filter(|l| !l.starts_with('#')).collect();
    let mut gpu_rows: Vec<&str> = gpu_content.lines().filter(|l| !l.starts_with('#')).collect();
    cpu_rows.sort();
    gpu_rows.sort();
    assert_eq!(
        cpu_rows, gpu_rows,
        "Myers CPU and GPU paths must produce identical matches"
    );

    // At least one match should exist for this known target in the test genome.
    assert!(!cpu_rows.is_empty(), "expected at least one bulge-aware match");

    // Each row should have exactly 8 tab-separated columns.
    for row in &cpu_rows {
        let cols: Vec<&str> = row.split('\t').collect();
        assert_eq!(
            cols.len(),
            8,
            "row does not have 8 columns: {:?}",
            row
        );
        // First column is bulge type.
        let t = cols[0];
        assert!(
            t == "X" || t == "DNA" || t == "RNA",
            "unexpected bulge type {:?}",
            t
        );
    }

    fs::remove_file(&in_path).ok();
    fs::remove_file(&cpu_out).ok();
    fs::remove_file(&gpu_out).ok();
    fs::remove_file(format!("{}.log", cpu_out.display())).ok();
    fs::remove_file(format!("{}.log", gpu_out.display())).ok();
}

// ---------------------------------------------------------------------------
// Log file is produced alongside output
// ---------------------------------------------------------------------------

#[test]
#[ignore]
fn integration_log_file_written() {
    let input_contents = format!(
        "{}\nNNNNNNNNNNNNNNNNNNNNNNNN\nCCGTGGTTCAACATTTGCTTAGCA 5\n",
        genome_path().display()
    );
    let in_path = make_temp_input(&input_contents, "log");
    let out_path = in_path.with_extension("out");
    let _ = run_cli(&in_path, "C", &out_path);
    let log_path = format!("{}.log", out_path.display());
    let log_content = fs::read_to_string(&log_path).expect(".log file should be created");
    assert!(log_content.contains("=== Cas-OFFinder Rust ==="));
    assert!(log_content.contains("Matches found:"));
    assert!(log_content.contains("Total elapsed:"));

    fs::remove_file(&in_path).ok();
    fs::remove_file(&out_path).ok();
    fs::remove_file(&log_path).ok();
}
