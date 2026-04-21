use crate::chrom_chunk::{ChromChunkInfo, CHUNK_SIZE, CHUNK_SIZE_BYTES};
use crate::cli_err::CliError;
use crate::string_to_bit4;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;
use std::sync::mpsc::SyncSender;

pub fn read_fasta(dest: &SyncSender<ChromChunkInfo>, fname: &Path) -> Result<(), CliError> {
    let file = File::open(fname)?;
    let buf_capacity = CHUNK_SIZE * 4;
    let buffer_reader = BufReader::with_capacity(buf_capacity, file);
    let mut started = false;
    let mut cur = ChromChunkInfo {
        chr_name: String::new(),
        chunk_start: 0,
        chunk_end: 0,
        data: Box::new([0_u8; CHUNK_SIZE_BYTES]),
    };
    for linerd in buffer_reader.lines() {
        let line = linerd?;
        if line.starts_with('>') {
            let next_chr_name = String::from_iter(line.chars().skip(1));
            let next_cur = ChromChunkInfo {
                chr_name: next_chr_name,
                chunk_start: 0,
                chunk_end: 0,
                data: Box::new([0_u8; CHUNK_SIZE_BYTES]),
            };
            if cur.chunk_end != cur.chunk_start {
                dest.send(cur)?;
            }
            cur = next_cur;
            started = true;
        } else {
            if !started {
                //catch this error to skip invalid files
                return Err(CliError::BadFileFormat("fasta file needs to start with >"));
            }
            if cur.chr_name.is_empty() {
                return Err(CliError::BadFileFormat(
                    "> must be followed by chromosome name",
                ));
            }
            if line.len() + cur.size() > CHUNK_SIZE {
                let next_cur = ChromChunkInfo {
                    chr_name: cur.chr_name.clone(),
                    chunk_start: cur.chunk_end,
                    chunk_end: cur.chunk_end,
                    data: Box::new([0_u8; CHUNK_SIZE_BYTES]),
                };
                dest.send(cur)?;
                cur = next_cur;
            }
            if line.len() > CHUNK_SIZE {
                return Err(CliError::BadFileFormat("line in fasta too long"));
            }
            let cur_size = cur.size() as usize;
            // mixed_base=true: keep genome 'N'/IUPAC chars distinguishable from
            // bit4=0 padding. N encodes as 0xF, so the Myers DP padding guard
            // (text_bit4 == 0) still works and the comparison logic can tell
            // "genome N" apart from "chromosome-boundary / buffer padding".
            string_to_bit4(&mut cur.data[..], line.as_bytes(), cur_size, true);
            cur.chunk_end += line.len() as u64;
        }
    }
    if cur.size() > 0 {
        dest.send(cur)?;
    }
    Ok(())
}

/*
unit tests for this in integration tests.
*/
