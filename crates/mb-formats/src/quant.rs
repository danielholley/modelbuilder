//! Encoders: f32 values to a tensor type's bytes, the inverse of
//! [`crate::dequant`]. Block formats follow ggml's deterministic reference
//! quantizers (`quantize_row_*_ref` in `ggml-quants.c`, and the PrismML
//! fork's for PQ2_0 and PTQ1_0), so re-encoding values that were decoded from
//! a file, or trained with the same fake-quantization, reproduces the file's
//! bytes.

use mb_ir::{DType, GgmlType};

#[derive(Debug, thiserror::Error)]
pub enum QuantError {
    #[error("no encoder for {0}")]
    Unsupported(DType),
    #[error("{dtype}: {n} values is not a whole number of {block}-value blocks")]
    Partial {
        dtype: DType,
        n: usize,
        block: usize,
    },
}

/// Round-to-nearest-even `f32` → BF16 bits (NaN stays NaN).
pub fn f32_to_bf16(x: f32) -> u16 {
    let bits = x.to_bits();
    if x.is_nan() {
        return ((bits >> 16) as u16) | 0x0040;
    }
    let round = 0x7fff + ((bits >> 16) & 1);
    (bits.wrapping_add(round) >> 16) as u16
}

/// Round-to-nearest-even `f32` → IEEE half bits (overflow saturates to ±inf).
pub fn f32_to_f16(x: f32) -> u16 {
    let bits = x.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32;
    let man = bits & 0x7f_ffff;
    if exp == 0xff {
        return sign | 0x7c00 | if man != 0 { 0x200 } else { 0 };
    }
    let e = exp - 127 + 15;
    if e >= 0x1f {
        return sign | 0x7c00;
    }
    if e <= 0 {
        if e < -10 {
            return sign;
        }
        // Subnormal: shift in the implicit bit, round to nearest even.
        let m = man | 0x80_0000;
        let shift = (14 - e) as u32;
        let half = 1u32 << (shift - 1);
        let rem = m & ((1 << shift) - 1);
        let mut v = m >> shift;
        if rem > half || (rem == half && (v & 1) == 1) {
            v += 1;
        }
        return sign | v as u16;
    }
    let mut v = ((e as u32) << 10) | (man >> 13);
    let rem = man & 0x1fff;
    if rem > 0x1000 || (rem == 0x1000 && (v & 1) == 1) {
        v += 1; // may carry into the exponent, which is correct rounding
    }
    sign | v as u16
}

fn f16_to_f32(h: u16) -> f32 {
    crate::dequant::half(&h.to_le_bytes())
}

fn blocks(
    dtype: DType,
    x: &[f32],
    block: usize,
) -> Result<std::slice::ChunksExact<'_, f32>, QuantError> {
    if x.len() % block != 0 {
        return Err(QuantError::Partial {
            dtype,
            n: x.len(),
            block,
        });
    }
    Ok(x.chunks_exact(block))
}

fn amax(b: &[f32]) -> f32 {
    b.iter().fold(0f32, |m, v| m.max(v.abs()))
}

/// Ternary code for `w` with inverse scale `id`: -1, 0, +1 → 0, 1, 2.
fn trit(w: f32, id: f32) -> u8 {
    ((w * id).round() as i32 + 1).clamp(0, 2) as u8
}

/// TQ1_0-style packing: 5 (or 4) trits, most significant first, scaled to a byte.
fn pack_trits(trits: impl Iterator<Item = u8>, shift: bool) -> u8 {
    let mut q: u16 = trits.fold(0, |q, t| q * 3 + u16::from(t));
    if shift {
        q *= 3; // `qh` bytes hold 4 trits: shift the first to the top
    }
    (q * 256).div_ceil(243) as u8
}

/// Appends the encoding of `x` as `dtype` to `out`.
pub fn quantize(dtype: DType, x: &[f32], out: &mut Vec<u8>) -> Result<(), QuantError> {
    match dtype {
        DType::F32 => out.extend(x.iter().flat_map(|v| v.to_le_bytes())),
        DType::F16 => out.extend(x.iter().flat_map(|v| f32_to_f16(*v).to_le_bytes())),
        DType::Bf16 => out.extend(x.iter().flat_map(|v| f32_to_bf16(*v).to_le_bytes())),
        DType::Ggml(GgmlType(8)) => {
            // Q8_0: { f16 d; i8 qs[32] }, d = amax / 127.
            for b in blocks(dtype, x, 32)? {
                let d = amax(b) / 127.0;
                let id = if d > 0.0 { 1.0 / d } else { 0.0 };
                out.extend(f32_to_f16(d).to_le_bytes());
                out.extend(b.iter().map(|v| (v * id).round() as i8 as u8));
            }
        }
        DType::Ggml(GgmlType(142)) => {
            // PQ2_0: { f16 d; u8 qs[32] }, d = amax, code = round(w/d) + 1.
            for b in blocks(dtype, x, 128)? {
                let d = amax(b);
                let id = if d > 0.0 { 1.0 / d } else { 0.0 };
                out.extend(f32_to_f16(d).to_le_bytes());
                for four in b.chunks_exact(4) {
                    out.push(
                        four.iter()
                            .enumerate()
                            .fold(0u8, |acc, (k, &w)| acc | (trit(w, id) << (2 * k))),
                    );
                }
            }
        }
        DType::Ggml(GgmlType(143)) => {
            // PTQ1_0: { u8 qs[24]; u8 qh[2]; f16 d }. `qs` in chunks of 16 then 8
            // bytes (element n*c + m is trit n of byte m), `qh` the last 8 weights.
            for b in blocks(dtype, x, 128)? {
                let d = amax(b);
                let id = if d > 0.0 { 1.0 / d } else { 0.0 };
                let mut at = 0;
                for c in [16usize, 8] {
                    let chunk = &b[at..at + 5 * c];
                    for m in 0..c {
                        out.push(pack_trits(
                            (0..5).map(|n| trit(chunk[m + n * c], id)),
                            false,
                        ));
                    }
                    at += 5 * c;
                }
                let tail = &b[at..];
                for h in 0..2 {
                    out.push(pack_trits((0..4).map(|n| trit(tail[h + n * 2], id)), true));
                }
                out.extend(f32_to_f16(d).to_le_bytes());
            }
        }
        other => return Err(QuantError::Unsupported(other)),
    }
    Ok(())
}

/// Encodes then decodes: the values a tensor of `dtype` can actually hold,
/// with the reference quantizer's choices. For fake-quantization and checks.
pub fn round_trip(dtype: DType, x: &[f32]) -> Result<Vec<f32>, QuantError> {
    let mut bytes = Vec::new();
    quantize(dtype, x, &mut bytes)?;
    let mut back = Vec::with_capacity(x.len());
    crate::dequant::dequantize(dtype, &bytes, &mut back)
        .map_err(|_| QuantError::Unsupported(dtype))?;
    Ok(back)
}

/// Whether [`quantize`] handles `dtype`.
pub fn supports(dtype: DType) -> bool {
    matches!(
        dtype,
        DType::F32
            | DType::F16
            | DType::Bf16
            | DType::Ggml(GgmlType(8))
            | DType::Ggml(GgmlType(142))
            | DType::Ggml(GgmlType(143))
    )
}

#[doc(hidden)]
pub fn _f16_roundtrip(x: f32) -> f32 {
    f16_to_f32(f32_to_f16(x))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dequant::dequantize;
    use crate::dequant_golden::{PQ2_0_HEX, PTQ1_0_HEX};

    fn unhex(s: &str) -> Vec<u8> {
        (0..s.len())
            .step_by(2)
            .map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap())
            .collect()
    }

    /// Golden blocks were written by the fork's own quantizer: decoding and
    /// re-encoding them must give the same bytes.
    #[test]
    fn re_encodes_the_forks_golden_blocks_byte_for_byte() {
        for (id, hex) in [(142u32, PQ2_0_HEX), (143, PTQ1_0_HEX)] {
            let dtype = DType::Ggml(GgmlType(id));
            let bytes = unhex(hex);
            let mut vals = Vec::new();
            dequantize(dtype, &bytes, &mut vals).unwrap();
            let mut again = Vec::new();
            quantize(dtype, &vals, &mut again).unwrap();
            assert_eq!(again, bytes, "type {id}");
        }
    }

    #[test]
    fn q8_0_and_floats_round_trip() {
        let x: Vec<f32> = (0..64).map(|i| ((i as f32) * 0.37).sin() * 3.0).collect();
        let q = round_trip(DType::Ggml(GgmlType(8)), &x).unwrap();
        let max_err = x
            .iter()
            .zip(&q)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(max_err <= 3.0 / 127.0, "{max_err}");
        assert_eq!(round_trip(DType::F32, &x).unwrap(), x);
        for v in [0.0f32, 1.0, -2.5, 65504.0, 1e-5, 6.1e-5, 2.71] {
            let h = _f16_roundtrip(v);
            assert!((h - v).abs() <= v.abs() * 1e-3 + 1e-7, "{v} → {h}");
        }
        assert_eq!(f32_to_f16(1.0), 0x3c00);
        assert_eq!(f32_to_f16(-2.0), 0xc000);
        assert_eq!(f32_to_f16(1e6), 0x7c00);
    }

    #[test]
    fn ternary_values_are_fixed_points() {
        // Values already of the form d·{-1,0,1} (per block) come back unchanged.
        let d = 0.0371f32;
        let d16 = f16_to_f32(f32_to_f16(d));
        let x: Vec<f32> = (0..256).map(|i| [-1.0, 0.0, 1.0][i % 3] * d16).collect();
        for id in [142u32, 143] {
            assert_eq!(
                round_trip(DType::Ggml(GgmlType(id)), &x).unwrap(),
                x,
                "type {id}"
            );
        }
        assert!(matches!(
            quantize(DType::Ggml(GgmlType(142)), &x[..100], &mut Vec::new()),
            Err(QuantError::Partial { .. })
        ));
    }
}
