use crate::cdiv;
use crate::chrom_chunk::{ChromChunkInfo, CHUNK_SIZE, CHUNK_SIZE_BYTES};
use crate::cli_err::{CliError, Result};
use crate::{bit2_to_bit4, memsetbit4};
use std::cmp::{max, min};
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;
use std::sync::mpsc::SyncSender;

fn read_u32(reader: &mut BufReader<std::fs::File>) -> Result<u32> {
    let mut buf = [0_u8; 4];
    reader.read_exact(&mut buf)?;
    unsafe { Ok(std::mem::transmute::<[u8; 4], u32>(buf)) }
}
fn read_u8(reader: &mut BufReader<std::fs::File>) -> Result<u8> {
    let mut buf = [0_u8; 1];
    reader.read_exact(&mut buf)?;
    Ok(buf[0])
}
fn read_str(reader: &mut BufReader<std::fs::File>, n_bytes: usize) -> Result<String> {
    let mut str_buf = vec![0_u8; n_bytes];
    reader.read_exact(&mut str_buf)?;
    Ok(String::from_utf8(str_buf)?)
}
fn read_intvec(reader: &mut BufReader<std::fs::File>, n_els: usize) -> Result<Vec<u32>> {
    let mut int_buf = Vec::with_capacity(n_els);
    for _ in 0..n_els {
        int_buf.push(read_u32(reader)?);
    }
    Ok(int_buf)
}
pub fn read_2bit(dest: &SyncSender<ChromChunkInfo>, fname: &Path) -> Result<()> {
    let file = File::open(fname)?;
    let buf_capacity = CHUNK_SIZE;
    let mut reader = BufReader::with_capacity(buf_capacity, file);
    let headerval = read_u32(&mut reader)?;
    if headerval != 0x1A412743 {
        return Err(CliError::BadFileFormat(".2bit file badly formatted header"));
    }
    let version_num = read_u32(&mut reader)?;
    if version_num != 0 {
        // Version should be 0
        return Err(CliError::BadFileFormat(
            "only supports version 0 of .2bit format",
        ));
    }
    let chrcnt = read_u32(&mut reader)?;
    reader.seek_relative(4)?; // skip reserved bits

    let mut chrom_names: Vec<String> = Vec::with_capacity(chrcnt as usize);
    for _ in 0..chrcnt {
        let len_chrname = read_u8(&mut reader)?;
        let chromname = read_str(&mut reader, len_chrname as usize)?;
        chrom_names.push(chromname);
        reader.seek_relative(4)?; // Absolute position of each sequence
    }
    for chrname in chrom_names.iter() {
        let chrlen = read_u32(&mut reader)? as usize;
        let nblockcnt = read_u32(&mut reader)? as usize;

        let nblockstart = read_intvec(&mut reader, nblockcnt)?;
        let nblocksizes = read_intvec(&mut reader, nblockcnt)?;
        let mut nblocks: Vec<(u32, u32)> = nblockstart
            .iter()
            .zip(nblocksizes.iter())
            .map(|(a1, a2)| (*a1, *a2))
            .collect();
        nblocks.sort_by_key(|(start, _size)| *start);

        let maskblockcnt = read_u32(&mut reader)?;
        // skip mask infos
        reader.seek_relative((maskblockcnt * 8 + 4) as i64)?;

        assert!(CHUNK_SIZE % 4 == 0);
        const NUCL_PER_BYTE: usize = 4;
        const RAW_BUF_LEN: usize = CHUNK_SIZE / NUCL_PER_BYTE;
        let mut raw_buf = [0_u8; RAW_BUF_LEN];
        let mut read_pos = 0;
        let mut block_mask_idx: i64 = 0;

        while read_pos < chrlen {
            let read_size = min(chrlen - read_pos, CHUNK_SIZE);
            reader.read_exact(&mut raw_buf[..cdiv(read_size, NUCL_PER_BYTE)])?;
            let mut chrdata = Box::new([0_u8; CHUNK_SIZE_BYTES]);
            bit2_to_bit4(&mut chrdata[..], &raw_buf, read_size);
            //go back one in case previous zone overlaps with current block
            block_mask_idx = max(block_mask_idx - 1, 0);
            while block_mask_idx < nblocks.len() as i64 {
                let (bstart, bsize) = nblocks[block_mask_idx as usize];
                let block_chunk_start = bstart as i64 - read_pos as i64;
                let block_chunk_end = (bstart + bsize) as i64 - read_pos as i64;
                if block_chunk_start > read_size as i64 {
                    break;
                }
                // N-blocks in 2bit files mark genome 'N'. Encode them as
                // bit4=0xF (all bits set) so the downstream search can tell
                // genome 'N' apart from chunk-boundary / trailing padding
                // (bit4=0), matching cas-offinder C++ semantics.
                memsetbit4(
                    &mut chrdata[..],
                    0xF,
                    max(0, block_chunk_start) as usize,
                    min(max(0, block_chunk_end) as usize, read_size),
                );
                block_mask_idx += 1;
            }
            dest.send(ChromChunkInfo {
                chr_name: chrname.clone(),
                chunk_start: read_pos as u64,
                chunk_end: (read_pos + read_size) as u64,
                data: chrdata,
            })?;
            read_pos += read_size;
        }
    }
    Ok(())
}

/*
unit tests for this in integration tests.
*/
