//! Log file generation for cas-offinder runs.
//!
//! Writes a human-readable summary of a run to `<output>.log` containing
//! input configuration, device info, match count, and elapsed time.

use std::fs::File;
use std::io::Write;
use std::path::Path;
use std::time::SystemTime;

/// Data captured for a single run, written out at completion.
pub struct RunLog {
    pub genome_path: String,
    pub genome_size: u64,
    pub n_patterns: usize,
    pub search_filter: String,
    pub max_mismatches: u32,
    pub max_dna_bulges: u32,
    pub max_rna_bulges: u32,
    pub device_label: String,
    pub algorithm: String, // "Myers bit-parallel" or "popcount (legacy)"
    pub n_matches: u64,
    pub total_elapsed_secs: f64,
}

/// Write the run log to the given path. Overwrites if exists.
pub fn write_log<P: AsRef<Path>>(log_path: P, log: &RunLog) -> std::io::Result<()> {
    let mut f = File::create(log_path)?;
    let secs = SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);

    writeln!(f, "=== Cas-OFFinder Rust ===")?;
    writeln!(f, "Run date (unix timestamp): {}", secs)?;
    writeln!(f)?;
    writeln!(f, "Algorithm: {}", log.algorithm)?;
    writeln!(f, "Device: {}", log.device_label)?;
    writeln!(f)?;
    writeln!(f, "Input:")?;
    writeln!(f, "  Genome: {}", log.genome_path)?;
    writeln!(f, "  Genome size: {} bytes", log.genome_size)?;
    writeln!(f, "  Patterns: {}", log.n_patterns)?;
    writeln!(f, "  Search filter: {}", log.search_filter)?;
    writeln!(f, "  Max mismatches: {}", log.max_mismatches)?;
    writeln!(f, "  Max DNA bulges: {}", log.max_dna_bulges)?;
    writeln!(f, "  Max RNA bulges: {}", log.max_rna_bulges)?;
    writeln!(f)?;
    writeln!(f, "Results:")?;
    writeln!(f, "  Matches found: {}", log.n_matches)?;
    writeln!(f, "  Total elapsed: {:.3}s", log.total_elapsed_secs)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_write_log_roundtrip() {
        let path = format!(
            "/tmp/test_log_{}_{}.log",
            std::process::id(),
            SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        );
        let log = RunLog {
            genome_path: "/tmp/genome.fa".to_string(),
            genome_size: 13_200,
            n_patterns: 2,
            search_filter: "NNNNNNNNNNNNNNNNNNNNNNGG".to_string(),
            max_mismatches: 5,
            max_dna_bulges: 1,
            max_rna_bulges: 1,
            device_label: "CPU (native Rust)".to_string(),
            algorithm: "Myers bit-parallel".to_string(),
            n_matches: 42,
            total_elapsed_secs: 1.234,
        };
        write_log(&path, &log).unwrap();
        let content = std::fs::read_to_string(&path).unwrap();
        assert!(content.contains("=== Cas-OFFinder Rust ==="));
        assert!(content.contains("Algorithm: Myers bit-parallel"));
        assert!(content.contains("Matches found: 42"));
        assert!(content.contains("Total elapsed: 1.234s"));
        assert!(content.contains("Max DNA bulges: 1"));
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_write_log_default_legacy() {
        let path = format!(
            "/tmp/test_log_legacy_{}_{}.log",
            std::process::id(),
            SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        );
        let log = RunLog {
            genome_path: "/x.fa".to_string(),
            genome_size: 0,
            n_patterns: 0,
            search_filter: String::new(),
            max_mismatches: 0,
            max_dna_bulges: 0,
            max_rna_bulges: 0,
            device_label: "GPU".to_string(),
            algorithm: "popcount (legacy)".to_string(),
            n_matches: 0,
            total_elapsed_secs: 0.0,
        };
        write_log(&path, &log).unwrap();
        let content = std::fs::read_to_string(&path).unwrap();
        assert!(content.contains("Algorithm: popcount (legacy)"));
        std::fs::remove_file(&path).ok();
    }
}
