//! Tensor decoding to `f32`, one tensor (or row) at a time.
//!
//! Block layouts for the PrismML vendor types follow `ggml/src/ggml-common.h`
//! and `ggml/src/ggml-quants.c` in PrismML-Eng/llama.cpp (branch `prism`,
//! checked at commit adfffbe); see `docs/research/prismml-quant-formats.md`.
//! Golden vectors in `dequant_golden.rs` come from that code's reference
//! quantizer, via `scripts/prism-golden.sh`.

use mb_ir::{DType, GgmlType};

#[derive(Debug, PartialEq, thiserror::Error)]
pub enum DequantError {
    #[error("decoding {0} is not supported")]
    Unsupported(DType),
    #[error("{dtype}: {bytes} bytes is not a whole number of {block_bytes}-byte blocks")]
    Misaligned {
        dtype: DType,
        bytes: usize,
        block_bytes: usize,
    },
}

/// IEEE 754 half precision to `f32`, including subnormals, infinities and NaN.
pub fn f16_to_f32(h: u16) -> f32 {
    let sign = u32::from(h >> 15) << 31;
    let exp = u32::from((h >> 10) & 0x1f);
    let mant = u32::from(h & 0x3ff);
    let bits = match (exp, mant) {
        (0, 0) => sign,
        (0, m) => {
            // Subnormal: normalize the mantissa.
            let shift = m.leading_zeros() - 21;
            sign | ((113 - shift) << 23) | (((m << shift) & 0x3ff) << 13)
        }
        (0x1f, m) => sign | 0x7f80_0000 | (m << 13),
        (e, m) => sign | ((e + 112) << 23) | (m << 13),
    };
    f32::from_bits(bits)
}

pub(crate) fn half(b: &[u8]) -> f32 {
    f16_to_f32(u16::from_le_bytes([b[0], b[1]]))
}

fn blocks(
    dtype: DType,
    data: &[u8],
    block_bytes: usize,
) -> Result<std::slice::ChunksExact<'_, u8>, DequantError> {
    if data.len() % block_bytes != 0 {
        return Err(DequantError::Misaligned {
            dtype,
            bytes: data.len(),
            block_bytes,
        });
    }
    Ok(data.chunks_exact(block_bytes))
}

/// PQ2_0 (type 142): `{ f16 d; u8 qs[32] }` per 128 weights. Element `j` is
/// bits `2*(j%4)..` of `qs[j/4]`; codes 0..3 decode to −1, 0, +1, +2 (times `d`).
/// Ternary checkpoints never use code 3.
fn pq2_0(data: &[u8], out: &mut Vec<f32>) -> Result<(), DequantError> {
    for b in blocks(DType::Ggml(GgmlType(142)), data, 34)? {
        let d = half(&b[0..2]);
        for j in 0..128 {
            let q = (b[2 + j / 4] >> ((j % 4) * 2)) & 0x03;
            out.push((i32::from(q) - 1) as f32 * d);
        }
    }
    Ok(())
}

/// Extracts trit `n` (most significant first) from a TQ1_0-style packed byte.
fn trit(byte: u8, n: usize) -> f32 {
    const POW3: [u8; 5] = [1, 3, 9, 27, 81];
    let q = byte.wrapping_mul(POW3[n]);
    ((u16::from(q) * 3) >> 8) as f32 - 1.0
}

/// PTQ1_0 (type 143): `{ u8 qs[24]; u8 qh[2]; f16 d }` per 128 weights.
/// `qs` holds 5 trits per byte in chunks of 16 then 8 bytes; within a chunk of
/// `c` bytes, element `n*c + m` is trit `n` of byte `m`. `qh` holds the last 8
/// weights, 4 trits per byte, element `n*2 + h` being trit `n` of `qh[h]`.
fn ptq1_0(data: &[u8], out: &mut Vec<f32>) -> Result<(), DequantError> {
    for b in blocks(DType::Ggml(GgmlType(143)), data, 28)? {
        let (qs, qh, d) = (&b[0..24], &b[24..26], half(&b[26..28]));
        for (start, c) in [(0usize, 16usize), (16, 8)] {
            for n in 0..5 {
                for m in 0..c {
                    out.push(trit(qs[start + m], n) * d);
                }
            }
        }
        for n in 0..4 {
            for &h in qh {
                out.push(trit(h, n) * d);
            }
        }
    }
    Ok(())
}

/// Q8_0 (type 8): `{ f16 d; i8 qs[32] }`.
fn q8_0(data: &[u8], out: &mut Vec<f32>) -> Result<(), DequantError> {
    for b in blocks(DType::Ggml(GgmlType(8)), data, 34)? {
        let d = half(&b[0..2]);
        out.extend(b[2..].iter().map(|&q| f32::from(q as i8) * d));
    }
    Ok(())
}

/// Appends the decoded values of `data` (a whole tensor or a whole number of
/// rows) to `out`.
pub fn dequantize(dtype: DType, data: &[u8], out: &mut Vec<f32>) -> Result<(), DequantError> {
    let plain = |width: usize, f: &dyn Fn(&[u8]) -> f32, out: &mut Vec<f32>| {
        blocks(dtype, data, width).map(|c| out.extend(c.map(f)))
    };
    match dtype {
        DType::F32 => plain(4, &|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]), out),
        DType::F16 => plain(2, &half, out),
        DType::Bf16 => plain(
            2,
            &|b| f32::from_bits(u32::from(u16::from_le_bytes([b[0], b[1]])) << 16),
            out,
        ),
        DType::Ggml(GgmlType(8)) => q8_0(data, out),
        DType::Ggml(GgmlType(142)) => pq2_0(data, out),
        DType::Ggml(GgmlType(143)) => ptq1_0(data, out),
        other => Err(DequantError::Unsupported(other)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dequant_golden::{CODES, PQ2_0_HEX, PTQ1_0_HEX, SCALES};

    fn unhex(s: &str) -> Vec<u8> {
        (0..s.len())
            .step_by(2)
            .map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap())
            .collect()
    }

    fn expected() -> Vec<f32> {
        CODES
            .iter()
            .enumerate()
            .map(|(i, &c)| f32::from(c) * SCALES[i / 128])
            .collect()
    }

    #[test]
    fn pq2_0_matches_prism_reference() {
        let mut out = Vec::new();
        dequantize(DType::Ggml(GgmlType(142)), &unhex(PQ2_0_HEX), &mut out).unwrap();
        assert_eq!(out, expected());
    }

    #[test]
    fn ptq1_0_matches_prism_reference() {
        let mut out = Vec::new();
        dequantize(DType::Ggml(GgmlType(143)), &unhex(PTQ1_0_HEX), &mut out).unwrap();
        assert_eq!(out, expected());
    }

    #[test]
    fn half_precision() {
        assert_eq!(f16_to_f32(0x3c00), 1.0);
        assert_eq!(f16_to_f32(0xc000), -2.0);
        assert_eq!(f16_to_f32(0x7bff), 65504.0);
        assert_eq!(f16_to_f32(0x0001), 2f32.powi(-24)); // smallest subnormal
        assert_eq!(f16_to_f32(0x03ff), 1023.0 * 2f32.powi(-24));
        assert!(f16_to_f32(0x7c00).is_infinite() && f16_to_f32(0x7e00).is_nan());
        assert_eq!(f16_to_f32(0x8000).to_bits(), (-0.0f32).to_bits());
    }

    #[test]
    fn plain_and_q8_0() {
        let mut out = Vec::new();
        dequantize(DType::Bf16, &[0x80, 0x3f, 0x00, 0xc0], &mut out).unwrap();
        assert_eq!(out, [1.0, -2.0]);
        let mut block = vec![0x00, 0x38]; // d = 0.5
        block.extend((0..32).map(|i| (i as i8 - 16) as u8));
        out.clear();
        dequantize(DType::Ggml(GgmlType(8)), &block, &mut out).unwrap();
        assert_eq!(out[0], -8.0);
        assert_eq!(out[31], 7.5);
    }

    #[test]
    fn rejects_partial_blocks_and_unknown_types() {
        let mut out = Vec::new();
        assert!(matches!(
            dequantize(DType::Ggml(GgmlType(142)), &[0; 33], &mut out),
            Err(DequantError::Misaligned { .. })
        ));
        assert!(matches!(
            dequantize(DType::Ggml(GgmlType(200)), &[0; 34], &mut out),
            Err(DequantError::Unsupported(_))
        ));
    }
}
