const T: u8 = 0x1;
const C: u8 = 0x2;
const A: u8 = 0x4;
const G: u8 = 0x8;

const NCHRS: usize = 1 << 8;
const NSHRTS: usize = 1 << 16;

pub fn cdiv(x: usize, y: usize) -> usize {
    (x + y - 1) / y
}

pub fn roundup(x: usize, y: usize) -> usize {
    cdiv(x, y) * y
}

const fn makebit4map(mixed_base: bool) -> [u8; NCHRS] {
    let mut arr = [0_u8; NCHRS];
    arr['G' as usize] = G;
    arr['C' as usize] = C;
    arr['A' as usize] = A;
    arr['T' as usize] = T;
    if mixed_base {
        arr['R' as usize] = A | G;
        arr['Y' as usize] = C | T;
        arr['S' as usize] = G | C;
        arr['W' as usize] = A | T;
        arr['K' as usize] = G | T;
        arr['M' as usize] = A | C;
        arr['B' as usize] = C | G | T;
        arr['D' as usize] = A | G | T;
        arr['H' as usize] = A | C | T;
        arr['V' as usize] = A | C | G;
        arr['N' as usize] = A | C | G | T;
    }
    arr
}
const fn apply_lower(inarr: [u8; NCHRS]) -> [u8; NCHRS] {
    let mut arr = [0_u8; NCHRS];
    let mut i = 1;
    while i <= 26 {
        arr[i + 96] = inarr[i + 64];
        arr[i + 64] = inarr[i + 64];
        i += 1;
    }
    arr
}
const fn invert_chrmap(inarr: [u8; NCHRS]) -> [u8; NCHRS] {
    let mut arr = [0_u8; NCHRS];
    let mut i = 0;
    while i < NCHRS {
        let idx = NCHRS - i - 1;
        if inarr[idx] != 0 {
            arr[inarr[idx] as usize] = idx as u8;
        }
        i += 1;
    }
    arr
}
const fn create_2bit_block_map() -> [u16; NCHRS] {
    let mut arr = [0_u16; NCHRS];
    let mut i = 0;
    while i < NCHRS {
        let mut val: u16 = 0;
        let mut j = 0;
        while j < 4 {
            let bit2val = (i >> ((4 - j - 1) * 2)) & 0x3;
            val |= (1 << bit2val) << (j * 4);
            j += 1;
        }
        arr[i] = val;
        i += 1;
    }
    arr
}
const fn doubleup_patternmap(inarr: [u8; NCHRS]) -> [u8; NSHRTS] {
    let mut arr = [0_u8; NSHRTS];
    let mut i = 0;
    let mut outidx = 0;
    while i < NCHRS {
        let mut j = 0;
        let shftval = inarr[i] << 4;
        while j < NCHRS {
            arr[outidx] = inarr[j] | shftval;
            outidx += 1;
            j += 1;
        }
        i += 1;
    }
    arr
}
const fn invert_double_patternmap(inarr: [u8; NSHRTS]) -> [u16; NCHRS] {
    let mut arr = [0_u16; NCHRS];
    let mut i = 0;
    while i < NSHRTS {
        let idx = NSHRTS - 1 - i;
        if inarr[idx] != 0 {
            arr[inarr[idx] as usize] = idx as u16;
        }
        i += 1;
    }
    arr
}

/*
Precomputed mappings between different data formats
*/

const STR_2_BIT4: [[u8; NCHRS]; 2] = [
    apply_lower(makebit4map(false)),
    apply_lower(makebit4map(true)),
];
const DSTR_TO_BIT4: [[u8; NSHRTS]; 2] = [
    doubleup_patternmap(STR_2_BIT4[0]),
    doubleup_patternmap(STR_2_BIT4[1]),
];

const BIT4_TO_STR: [u8; NCHRS] = invert_chrmap(STR_2_BIT4[1]);
const DBIT4_TO_STR: [u16; NCHRS] = invert_double_patternmap(DSTR_TO_BIT4[1]);

const BIT2_TO_BIT4: [u16; 256] = create_2bit_block_map();

pub fn bit4_to_string(out_data: &mut [u8], data: &[u8], read_offset: usize, n_chrs: usize) {
    assert!(
        out_data.len() >= n_chrs,
        "out data must be large enough to store result"
    );
    let src = &data[(read_offset / 2)..];
    let r_offset = read_offset % 2;
    if r_offset != 0 && n_chrs > 0 {
        out_data[0] = BIT4_TO_STR[((src[0] & 0xf0) >> 4) as usize];
        bit4_to_string(&mut out_data[1..], &src[1..], 0, n_chrs - 1);
    } else {
        unsafe {
            let dout_data_ptr = out_data.as_ptr() as *mut u16;
            for i in 0..n_chrs / 2 {
                *dout_data_ptr.add(i) = DBIT4_TO_STR[*src.get_unchecked(i) as usize];
            }
        }
        if n_chrs % 2 == 1 {
            out_data[n_chrs - 1] = BIT4_TO_STR[(src[n_chrs / 2] & 0x0f) as usize];
        }
    }
}
pub fn string_to_bit4(out_data: &mut [u8], data: &[u8], write_offset: usize, mixed_base: bool) {
    let n_chrs = data.len();
    let dest = &mut out_data[write_offset / 2..];
    if write_offset % 2 != 0 && n_chrs > 0 {
        dest[0] |= STR_2_BIT4[mixed_base as usize][data[0] as usize] << 4;
        string_to_bit4(&mut dest[1..], &data[1..], 0, mixed_base);
    } else {
        assert!(dest.len() >= cdiv(n_chrs, 2));
        if n_chrs % 2 != 0 {
            dest[n_chrs / 2] |= STR_2_BIT4[mixed_base as usize][data[n_chrs - 1] as usize];
        }
        unsafe {
            let srcptr = data.as_ptr();
            let dsrcptr = srcptr as *const u16;
            for i in 0..(n_chrs / 2) {
                *dest.get_unchecked_mut(i) =
                    DSTR_TO_BIT4[mixed_base as usize][(*(dsrcptr.add(i))) as usize];
            }
        }
    }
}
pub fn bit2_to_bit4(out_data: &mut [u8], data: &[u8], n_chrs: usize) {
    assert!(
        out_data.as_ptr().align_offset(2) == 0,
        "dest must be alligned to 2 byte boundaries"
    );
    let n_blks = cdiv(n_chrs, 4);
    assert!(out_data.len() * 2 >= n_blks * 4);
    assert!(data.len() * 4 >= n_blks * 4);
    unsafe {
        let blkdest = out_data.as_mut_ptr() as *mut u16;
        for i in 0..n_blks {
            *blkdest.add(i) = BIT2_TO_BIT4[*data.get_unchecked(i) as usize];
        }
    }
    if n_chrs % 4 != 0 {
        memsetbit4(out_data, 0, n_chrs, n_blks * 4);
    }
}
pub fn memsetbit4(dest: &mut [u8], bit4val: u8, start: usize, end: usize) {
    assert!(bit4val <= 0xf);
    if start < end && start % 2 == 1 {
        dest[start / 2] = (dest[start / 2] & 0xf) | (bit4val << 4);
        memsetbit4(dest, bit4val, start + 1, end);
    } else if start < end && end % 2 == 1 {
        dest[end / 2] = (dest[end / 2] & 0xf0) | bit4val;
        memsetbit4(dest, bit4val, start, end - 1);
    } else if start < end {
        let bstart = start / 2;
        let bend = end / 2;
        let bval = bit4val | (bit4val << 4);
        for item in &mut dest[bstart..bend] {
            *item = bval;
        }
    }
}
pub fn is_mixedbase(c: u8) -> bool {
    STR_2_BIT4[true as usize][c as usize] != 0
}
pub fn is_mixedbase_str(chars: &[u8]) -> bool {
    chars.iter().copied().map(is_mixedbase).all(|x| x)
}
fn complimentb4(v: u8) -> u8 {
    // only operates on the first 4 bits
    ((v << 2) | (v >> 2)) & 0xf
}
fn compliment_char(c: u8) -> u8 {
    let b4 = STR_2_BIT4[true as usize][c as usize];
    let rev_bit4 = complimentb4(b4);
    let rev_char = if b4 != 0 {
        BIT4_TO_STR[rev_bit4 as usize]
    } else {
        c
    };
    //use original capitalization
    rev_char | (c & !0xdf)
}
pub fn reverse_compliment_char_i(out_data: &mut [u8]) {
    for c in out_data.iter_mut() {
        *c = compliment_char(*c);
    }
    out_data.reverse();
}
pub fn reverse_compliment_char(out_data: &[u8]) -> Vec<u8> {
    let mut res: Vec<u8> = out_data.to_vec();
    reverse_compliment_char_i(&mut res);
    res
}
pub fn cmp_chars(dna: u8, rna: u8) -> bool {
    let dnab = STR_2_BIT4[false as usize][dna as usize];
    let rnab = STR_2_BIT4[true as usize][rna as usize];
    (dnab & rnab) != 0
}

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

/// Map a single ASCII character to its 4-bit nucleotide encoding.
/// N/mixed bases expand to union of bits.
pub fn char_to_bit4(c: u8) -> u8 {
    STR_2_BIT4[true as usize][c as usize]
}

/// Read the bit4 nucleotide at nucleotide index `i` from a packed buffer.
/// Two nucleotides per byte: even index in the low nibble, odd in the high.
#[inline(always)]
pub fn get_bit4(data: &[u8], i: usize) -> u8 {
    let byte = data[i / 2];
    if i % 2 == 0 {
        byte & 0x0F
    } else {
        (byte >> 4) & 0x0F
    }
}

/// Inverse of `char_to_bit4`: map a 4-bit code back to an ASCII character
/// (uppercase IUPAC if mixed-base, else 'N'/'0' for invalid).
#[inline(always)]
pub fn bit4_to_char(b: u8) -> u8 {
    BIT4_TO_STR[(b & 0x0F) as usize]
}

