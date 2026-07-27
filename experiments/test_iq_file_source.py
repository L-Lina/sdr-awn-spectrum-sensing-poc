"""
Unit tests for src/io/iq_file_source.py -- the formal, reusable IQ file
loader wired into src/utils/pipeline.py's cfile branch.

All fixtures are generated on the fly into a tempfile.TemporaryDirectory
and cleaned up automatically; nothing is written into the formal dataset
or committed to git. Run directly:

    python experiments/test_iq_file_source.py

Covers (numbered to match the 16 required verification items):
 1. complex64 correct load
 2. float32 interleaved correct reconstruction
 3. int16 interleaved correct reconstruction via scale
 4. complex64 ≈ float32 within float tolerance (bit-identical, in fact)
 5. int16 ≈ original within reasonable quantization error
 6. offset_samples correct
 7. max_samples correct
 8. little-endian correct
 9. big-endian correct
10. odd interleaved value count correctly rejected
11. empty file correctly rejected
12. nonexistent file correctly rejected
13. illegal format correctly rejected
14. illegal channel count correctly rejected
15. no NaN/Inf in output (and NaN/Inf-producing files correctly rejected)
16. output dtype is complex64, SHA256/file size/sample count metadata correct
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.io.iq_file_source import load_iq_file, write_iq_file, DEFAULT_INT16_SCALE  # noqa: E402

PASS = []
FAIL = []


def check(name: str, cond: bool, detail: str = "") -> None:
    status = "PASS" if cond else "FAIL"
    (PASS if cond else FAIL).append(name)
    print(f"[{status}] {name}" + (f" -- {detail}" if detail and not cond else ""))


def make_stream(n: int = 500, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    real = rng.normal(0, 0.05, n).astype(np.float32)
    imag = rng.normal(0, 0.05, n).astype(np.float32)
    return (real + 1j * imag).astype(np.complex64)


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="iq_file_source_test_") as tmp:
        tmp = Path(tmp)
        iq = make_stream(500)

        # ---- 1/4/16: complex64 correct load + provenance ----
        p_c64 = tmp / "s.c64"
        write_iq_file(iq, str(p_c64), "complex64")
        loaded_c64, prov_c64 = load_iq_file(str(p_c64), "complex64")
        check("1. complex64 correct load (shape+dtype)", loaded_c64.shape == iq.shape and loaded_c64.dtype == np.complex64)
        check("1b. complex64 correct load (values exact)", np.array_equal(loaded_c64, iq))
        check("16. dtype is complex64", str(loaded_c64.dtype) == "complex64")
        check("16b. file_size_bytes matches", prov_c64["file_size_bytes"] == p_c64.stat().st_size)
        import hashlib
        expected_sha = hashlib.sha256(p_c64.read_bytes()).hexdigest()
        check("16c. file_sha256 matches", prov_c64["file_sha256"] == expected_sha)
        check("16d. loaded_sample_count matches", prov_c64["loaded_sample_count"] == len(iq))

        # ---- 2/4: interleaved_float32 correct reconstruction ----
        p_f32 = tmp / "s.f32iq"
        write_iq_file(iq, str(p_f32), "interleaved_float32")
        loaded_f32, prov_f32 = load_iq_file(str(p_f32), "interleaved_float32")
        check("2. float32 interleaved correct reconstruction (shape)", loaded_f32.shape == iq.shape)
        max_diff_f32 = float(np.max(np.abs(loaded_f32 - iq)))
        check("4. complex64 ≈ float32 within tolerance", max_diff_f32 < 1e-6, f"max_diff={max_diff_f32}")

        # ---- 3/5: interleaved_int16 correct reconstruction via scale ----
        p_i16 = tmp / "s.i16iq"
        write_iq_file(iq, str(p_i16), "interleaved_int16")
        loaded_i16, prov_i16 = load_iq_file(str(p_i16), "interleaved_int16")
        check("3. int16 interleaved correct reconstruction (shape)", loaded_i16.shape == iq.shape)
        check("3b. int16 default scale recorded == DEFAULT_INT16_SCALE", prov_i16["scale"] == DEFAULT_INT16_SCALE)
        max_diff_i16 = float(np.max(np.abs(loaded_i16 - iq)))
        expected_quant_bound = 2.0 * DEFAULT_INT16_SCALE  # +-0.5 LSB per I and Q, generous margin
        check("5. int16 ≈ original within quantization tolerance", max_diff_i16 < expected_quant_bound,
              f"max_diff={max_diff_i16}, bound={expected_quant_bound}")

        # explicit custom scale
        custom_scale = 0.01
        p_i16b = tmp / "s_customscale.i16iq"
        write_iq_file(iq, str(p_i16b), "interleaved_int16", scale=custom_scale)
        _, prov_i16b = load_iq_file(str(p_i16b), "interleaved_int16", scale=custom_scale)
        check("3c. explicit int16 scale correctly recorded (not silently defaulted)", prov_i16b["scale"] == custom_scale)

        # ---- 6: offset_samples correct ----
        offset = 50
        loaded_off, prov_off = load_iq_file(str(p_c64), "complex64", offset_samples=offset)
        check("6. offset_samples correct (length)", len(loaded_off) == len(iq) - offset)
        check("6b. offset_samples correct (values)", np.array_equal(loaded_off, iq[offset:]))
        check("6c. offset_samples recorded in provenance", prov_off["offset_samples"] == offset)

        # ---- 7: max_samples correct ----
        maxs = 37
        loaded_max, prov_max = load_iq_file(str(p_c64), "complex64", max_samples=maxs)
        check("7. max_samples correct (length)", len(loaded_max) == maxs)
        check("7b. max_samples correct (values)", np.array_equal(loaded_max, iq[:maxs]))
        loaded_offmax, _ = load_iq_file(str(p_c64), "complex64", offset_samples=offset, max_samples=maxs)
        check("7c. offset+max_samples combined correct", np.array_equal(loaded_offmax, iq[offset:offset + maxs]))

        # ---- 8/9: endianness ----
        p_c64_le = tmp / "s_le.c64"
        write_iq_file(iq, str(p_c64_le), "complex64", endianness="little")
        loaded_le, prov_le = load_iq_file(str(p_c64_le), "complex64", endianness="little")
        check("8. little-endian correct", np.array_equal(loaded_le, iq) and prov_le["endianness"] == "little")

        p_c64_be = tmp / "s_be.c64"
        write_iq_file(iq, str(p_c64_be), "complex64", endianness="big")
        loaded_be, prov_be = load_iq_file(str(p_c64_be), "complex64", endianness="big")
        check("9. big-endian correct", np.array_equal(loaded_be, iq) and prov_be["endianness"] == "big")

        # cross-check: reading a big-endian file as little-endian must NOT match (sanity on the
        # test itself) -- byte-swapped float32 garbage frequently decodes to NaN/Inf, which the
        # loader correctly rejects; either a raised error or a value mismatch proves endianness
        # is genuinely applied, not ignored.
        try:
            loaded_be_as_le, _ = load_iq_file(str(p_c64_be), "complex64", endianness="little")
            check("9b. endianness actually affects decoding (mismatch when misread)", not np.array_equal(loaded_be_as_le, iq))
        except ValueError:
            check("9b. endianness actually affects decoding (mismatch when misread)", True,
                  "misread correctly produced non-finite values, rejected by loader")

        # ---- 10: odd interleaved value count correctly rejected ----
        p_odd = tmp / "odd.f32iq"
        odd_vals = np.arange(11, dtype=np.float32)  # 11 is odd
        odd_vals.tofile(p_odd)
        try:
            load_iq_file(str(p_odd), "interleaved_float32")
            check("10. odd interleaved count rejected (float32)", False)
        except ValueError:
            check("10. odd interleaved count rejected (float32)", True)

        p_odd16 = tmp / "odd.i16iq"
        np.arange(9, dtype=np.int16).tofile(p_odd16)
        try:
            load_iq_file(str(p_odd16), "interleaved_int16")
            check("10b. odd interleaved count rejected (int16)", False)
        except ValueError:
            check("10b. odd interleaved count rejected (int16)", True)

        # ---- 11: empty file correctly rejected ----
        p_empty = tmp / "empty.c64"
        p_empty.touch()
        try:
            load_iq_file(str(p_empty), "complex64")
            check("11. empty file rejected", False)
        except ValueError:
            check("11. empty file rejected", True)

        # ---- 12: nonexistent file correctly rejected ----
        try:
            load_iq_file(str(tmp / "does_not_exist.c64"), "complex64")
            check("12. nonexistent file rejected", False)
        except FileNotFoundError:
            check("12. nonexistent file rejected", True)

        # ---- 13: illegal format correctly rejected ----
        try:
            load_iq_file(str(p_c64), "not_a_real_format")
            check("13. illegal iq_format rejected", False)
        except ValueError:
            check("13. illegal iq_format rejected", True)

        try:
            load_iq_file(str(p_c64), "complex64", endianness="not_a_real_endian")
            check("13b. illegal endianness rejected", False)
        except ValueError:
            check("13b. illegal endianness rejected", True)

        # ---- 14: illegal channel count correctly rejected ----
        try:
            load_iq_file(str(p_c64), "complex64", channel_count=2)
            check("14. illegal channel_count rejected", False)
        except ValueError:
            check("14. illegal channel_count rejected", True)

        # additional boundary checks (offset<0, max_samples<=0, offset beyond file length)
        try:
            load_iq_file(str(p_c64), "complex64", offset_samples=-1)
            check("6d. negative offset_samples rejected", False)
        except ValueError:
            check("6d. negative offset_samples rejected", True)

        try:
            load_iq_file(str(p_c64), "complex64", max_samples=0)
            check("7d. max_samples<=0 rejected", False)
        except ValueError:
            check("7d. max_samples<=0 rejected", True)

        try:
            load_iq_file(str(p_c64), "complex64", offset_samples=len(iq) + 1)
            check("6e. offset beyond file length rejected", False)
        except ValueError:
            check("6e. offset beyond file length rejected", True)

        # ---- 15: no NaN/Inf in output; NaN/Inf-producing files rejected ----
        check("15. normal load has no NaN/Inf", bool(np.isfinite(loaded_c64).all()))
        p_nan = tmp / "nan.c64"
        bad = iq.copy()
        bad[3] = complex(np.nan, 0.0)
        bad.tofile(p_nan)
        try:
            load_iq_file(str(p_nan), "complex64")
            check("15b. NaN-containing file rejected", False)
        except ValueError:
            check("15b. NaN-containing file rejected", True)

        p_inf = tmp / "inf.c64"
        bad2 = iq.copy()
        bad2[3] = complex(np.inf, 0.0)
        bad2.tofile(p_inf)
        try:
            load_iq_file(str(p_inf), "complex64")
            check("15c. Inf-containing file rejected", False)
        except ValueError:
            check("15c. Inf-containing file rejected", True)

    print(f"\n{len(PASS)} PASS, {len(FAIL)} FAIL out of {len(PASS) + len(FAIL)}")
    if FAIL:
        print("FAILED:", FAIL)
        sys.exit(1)


if __name__ == "__main__":
    main()
