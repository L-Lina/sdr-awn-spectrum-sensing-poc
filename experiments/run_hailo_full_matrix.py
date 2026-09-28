from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.adapters.attack_adapter import AttackAdapter
from src.adapters.awn_adapter import AWNModelAdapter
from src.adapters.hailo_awn_adapter import HailoAWNAdapter
from src.adapters.topk_adapter import TopKAdapter
from src.sensing.energy_detection import (
    energy_detect,
    filter_by_min_length,
    mask_to_regions,
    merge_close_regions,
)
from src.sensing.ground_truth_metrics import compute_sensing_ground_truth_metrics
from src.sensing.normalize import apply_awn_preprocess, to_awn_input
from src.sensing.radioml_source import (
    RML2016_10A_CLASSES,
    embed_sample_in_noise,
    load_radioml_dict,
)
from src.sensing.segmentation import select_aligned_segments


# ---------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_array(x: np.ndarray) -> str:
    return hashlib.sha256(
        np.ascontiguousarray(x).tobytes()
    ).hexdigest()


def derive_seed(
    base_seed: int,
    mod: str,
    snr: int,
    sample_index: int,
    salt: str,
) -> int:
    s = f"{base_seed}|{mod}|{snr}|{sample_index}|{salt}"
    d = hashlib.sha256(s.encode()).digest()
    return int.from_bytes(d[:4], "big") % (2**31)


def is_bad_meta(meta: Dict[str, Any], prefix: str) -> bool:
    status = str(meta.get(f"{prefix}_status", "")).lower()
    backend = str(meta.get(f"{prefix}_backend", "")).lower()
    notes = str(meta.get(f"{prefix}_notes", "")).lower()

    return (
        status != "ok"
        or "dummy" in backend
        or "fallback" in status
        or "fallback" in backend
        or "fallback" in notes
    )


def elapsed_ms(t0: float) -> float:
    return (time.perf_counter() - t0) * 1000.0


def csv_append(path: Path, row: Dict[str, Any], fields: List[str]) -> None:
    new_file = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if new_file:
            w.writeheader()
        w.writerow(row)
        f.flush()
        os.fsync(f.fileno())


def read_done_keys(path: Path, fields: List[str]) -> set:
    if not path.exists() or path.stat().st_size == 0:
        return set()

    out = set()

    with path.open(newline="") as f:
        for r in csv.DictReader(f):
            out.add(tuple(str(r[x]) for x in fields))

    return out


def numeric_summary(values: Iterable[float]) -> Dict[str, float]:
    a = np.asarray(list(values), dtype=np.float64)

    if a.size == 0:
        return {
            "n": 0,
            "mean": math.nan,
            "median": math.nan,
            "p95": math.nan,
            "p99": math.nan,
            "min": math.nan,
            "max": math.nan,
            "std": math.nan,
        }

    return {
        "n": int(a.size),
        "mean": float(np.mean(a)),
        "median": float(np.median(a)),
        "p95": float(np.percentile(a, 95)),
        "p99": float(np.percentile(a, 99)),
        "min": float(np.min(a)),
        "max": float(np.max(a)),
        "std": float(np.std(a)),
    }


# ---------------------------------------------------------------------
# compatibility wrappers around existing sensing implementation
# ---------------------------------------------------------------------

def _required_without_default(param: inspect.Parameter) -> bool:
    return (
        param.default is inspect._empty
        and param.kind
        not in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        )
    )


def call_embed(
    sample: np.ndarray,
    n_samples: int,
    margin: float,
    seed: int,
):
    fn = embed_sample_in_noise
    sig = inspect.signature(fn)
    params = list(sig.parameters.values())

    args = [sample]
    kwargs: Dict[str, Any] = {}

    for p in params[1:]:
        n = p.name

        if n in ("n_samples", "stream_length", "stream_len"):
            kwargs[n] = n_samples
        elif n in (
            "snr_margin",
            "snr_margin_db",
            "embed_snr_margin",
            "margin",
            "margin_db",
        ):
            kwargs[n] = margin
        elif n == "seed":
            kwargs[n] = seed
        elif _required_without_default(p):
            raise RuntimeError(
                "Unsupported required embed_sample_in_noise parameter: "
                f"{n}; signature={sig}"
            )

    return fn(*args, **kwargs)


def call_merge(regions, gap: int):
    sig = inspect.signature(merge_close_regions)
    params = list(sig.parameters.values())

    if len(params) <= 1:
        return merge_close_regions(regions)

    p = params[1].name

    if p in ("gap", "merge_gap", "max_gap"):
        return merge_close_regions(regions, **{p: gap})

    return merge_close_regions(regions, gap)


def call_filter(regions, min_len: int):
    sig = inspect.signature(filter_by_min_length)
    params = list(sig.parameters.values())

    if len(params) <= 1:
        return filter_by_min_length(regions)

    p = params[1].name

    if p in (
        "min_len",
        "min_length",
        "min_region_len",
        "minimum_length",
    ):
        return filter_by_min_length(regions, **{p: min_len})

    return filter_by_min_length(regions, min_len)


def call_select(
    iq: np.ndarray,
    regions,
    seg_len: int,
    policy: str,
    hop: int,
):
    fn = select_aligned_segments
    sig = inspect.signature(fn)

    kwargs: Dict[str, Any] = {}

    for i, p in enumerate(sig.parameters.values()):
        n = p.name

        if i == 0:
            continue

        if n in ("regions", "occupied_regions"):
            kwargs[n] = regions
        elif n in (
            "seg_len",
            "segment_len",
            "window_size",
            "length",
        ):
            kwargs[n] = seg_len
        elif n in ("policy", "alignment_policy"):
            kwargs[n] = policy
        elif n in ("hop", "segment_hop", "step"):
            kwargs[n] = hop
        elif _required_without_default(p):
            raise RuntimeError(
                "Unsupported required select_aligned_segments "
                f"parameter: {n}; signature={sig}"
            )

    result = fn(iq, **kwargs)

    if not isinstance(result, tuple) or len(result) != 2:
        raise RuntimeError(
            "select_aligned_segments must return "
            "(segments, alignment_meta)"
        )

    return result


# ---------------------------------------------------------------------
# one base sample -> sensing input
# ---------------------------------------------------------------------

def build_sensing_input(
    sample: np.ndarray,
    cfg: Dict[str, Any],
    mod: str,
    snr: int,
    sample_index: int,
):
    s = cfg["sensing"]

    timing: Dict[str, float] = {}

    base_seed = int(s["base_seed"])

    seed = derive_seed(
        base_seed,
        mod,
        snr,
        sample_index,
        "",
    )

    t0 = time.perf_counter()

    iq_long, embed_meta = call_embed(
        sample=sample,
        n_samples=int(s["n_samples"]),
        margin=float(s["embed_snr_margin"]),
        seed=seed,
    )

    timing["embedding_ms"] = elapsed_ms(t0)

    if not np.isfinite(iq_long).all():
        raise RuntimeError("embedded IQ contains NaN/Inf")

    t0 = time.perf_counter()

    mask = energy_detect(
        iq_long,
        window=int(s["sensing_window_size"]),
        threshold_factor=float(s["threshold_factor"]),
    )

    timing["energy_detection_ms"] = elapsed_ms(t0)

    t0 = time.perf_counter()

    regions = mask_to_regions(mask)
    regions = call_merge(
        regions,
        int(s["merge_gap"]),
    )
    regions = call_filter(
        regions,
        int(s["min_region_len"]),
    )

    timing["region_postprocess_ms"] = elapsed_ms(t0)

    if not regions:
        raise RuntimeError(
            f"sensing produced zero regions for "
            f"{mod} snr={snr} idx={sample_index}"
        )

    t0 = time.perf_counter()

    segments, alignment_meta = call_select(
        iq_long,
        regions,
        seg_len=int(s["segment_len"]),
        policy=str(s["alignment_policy"]),
        hop=int(s["segment_hop"]),
    )

    timing["segmentation_alignment_ms"] = elapsed_ms(t0)

    if len(segments) == 0:
        raise RuntimeError(
            f"segmentation produced zero segments for "
            f"{mod} snr={snr} idx={sample_index}"
        )

    # existing formal convention: first region / first selected segment
    seg = np.asarray(segments[0], dtype=np.complex64)

    if seg.shape != (int(s["segment_len"]),):
        raise RuntimeError(
            f"unexpected selected segment shape {seg.shape}"
        )

    t0 = time.perf_counter()

    segs = seg[np.newaxis, :]
    segs = apply_awn_preprocess(
        segs,
        policy=str(s["awn_preprocess"]),
    )

    x_clean = to_awn_input(
        segs,
        seg_len=int(s["segment_len"]),
    ).astype(np.float32)

    timing["awn_preprocess_ms"] = elapsed_ms(t0)

    if x_clean.shape != (1, 2, int(s["segment_len"])):
        raise RuntimeError(
            f"unexpected AWN input shape {x_clean.shape}"
        )

    if not np.isfinite(x_clean).all():
        raise RuntimeError("clean AWN input contains NaN/Inf")

    true_start = int(embed_meta["true_start"])
    true_end = int(embed_meta["true_end"])

    gt = compute_sensing_ground_truth_metrics(
        true_start,
        true_end,
        regions,
    )

    am = None

    if isinstance(alignment_meta, list) and alignment_meta:
        am = alignment_meta[0]
    elif isinstance(alignment_meta, dict):
        am = alignment_meta
    else:
        am = {}

    meta = {
        "seed": seed,
        "true_start": true_start,
        "true_end": true_end,
        "region_count": len(regions),
        "selected_segment_start": am.get(
            "selected_segment_start"
        ),
        "selected_segment_end": am.get(
            "selected_segment_end"
        ),
        "detected_region_start": am.get(
            "detected_region_start"
        ),
        "detected_region_end": am.get(
            "detected_region_end"
        ),
        "captured_signal_ratio": gt.get(
            "captured_signal_ratio"
        ),
        "start_boundary_error": gt.get(
            "start_boundary_error"
        ),
        "end_boundary_error": gt.get(
            "end_boundary_error"
        ),
        "missed_signal_samples": gt.get(
            "missed_signal_samples"
        ),
        "false_occupied_samples": gt.get(
            "false_occupied_samples"
        ),
        "clean_input_sha256": sha256_array(x_clean),
    }

    return x_clean, meta, timing


# ---------------------------------------------------------------------
# aggregate
# ---------------------------------------------------------------------

def build_summary(
    out_dir: Path,
    attack_csv: Path,
    defense_csv: Path,
    base_csv: Path,
):
    summary: Dict[str, Any] = {}

    def read_csv(path):
        with path.open(newline="") as f:
            return list(csv.DictReader(f))

    base = read_csv(base_csv)
    attacks = read_csv(attack_csv)
    defenses = read_csv(defense_csv)

    summary["counts"] = {
        "base_rows": len(base),
        "attack_rows": len(attacks),
        "defense_rows": len(defenses),
    }

    latency_fields = [
        "embedding_ms",
        "energy_detection_ms",
        "region_postprocess_ms",
        "segmentation_alignment_ms",
        "awn_preprocess_ms",
        "clean_hailo_inference_ms",
        "attack_generation_ms",
        "attacked_hailo_inference_ms",
        "topk_transform_ms",
        "defended_hailo_inference_ms",
        "attack_pipeline_ms",
        "defense_pipeline_ms",
    ]

    summary["latency"] = {}

    for field in latency_fields:
        src = (
            base
            if field in base[0]
            else attacks
            if attacks and field in attacks[0]
            else defenses
        )

        vals = []

        for r in src:
            try:
                vals.append(float(r[field]))
            except (KeyError, ValueError, TypeError):
                pass

        summary["latency"][field] = numeric_summary(vals)

    attack_groups: Dict[str, List[dict]] = {}

    for r in attacks:
        k = (
            f"{r['attack']}|{r['attack_profile']}"
        )
        attack_groups.setdefault(k, []).append(r)

    summary["attacks"] = {}

    for k, rows in attack_groups.items():
        clean_correct = np.asarray(
            [r["clean_correct"] == "True" for r in rows],
            dtype=bool,
        )

        attacked_correct = np.asarray(
            [r["attacked_correct"] == "True" for r in rows],
            dtype=bool,
        )

        eligible = clean_correct

        success = (
            eligible
            & (~attacked_correct)
        )

        summary["attacks"][k] = {
            "n": len(rows),
            "clean_accuracy": float(
                np.mean(clean_correct)
            ),
            "attacked_accuracy": float(
                np.mean(attacked_correct)
            ),
            "conditional_asr": (
                float(np.mean(success[eligible]))
                if np.any(eligible)
                else math.nan
            ),
            "iq_linf_mean": float(
                np.mean([
                    float(r["iq_linf"])
                    for r in rows
                ])
            ),
            "iq_l2_mean": float(
                np.mean([
                    float(r["iq_l2"])
                    for r in rows
                ])
            ),
        }

    defense_groups: Dict[str, List[dict]] = {}

    for r in defenses:
        k = (
            f"{r['attack']}|"
            f"{r['attack_profile']}|"
            f"K={r['topk']}"
        )
        defense_groups.setdefault(k, []).append(r)

    summary["defenses"] = {}

    for k, rows in defense_groups.items():
        defended_correct = np.asarray(
            [r["defended_correct"] == "True" for r in rows],
            dtype=bool,
        )

        attack_success = np.asarray(
            [r["attack_success"] == "True" for r in rows],
            dtype=bool,
        )

        recovered = np.asarray(
            [r["recovered"] == "True" for r in rows],
            dtype=bool,
        )

        clean_correct = np.asarray(
            [r["clean_correct"] == "True" for r in rows],
            dtype=bool,
        )

        clean_topk_correct = np.asarray(
            [r["clean_topk_correct"] == "True" for r in rows],
            dtype=bool,
        )

        summary["defenses"][k] = {
            "n": len(rows),
            "defended_accuracy": float(
                np.mean(defended_correct)
            ),
            "recovery_rate_given_attack_success": (
                float(np.mean(recovered[attack_success]))
                if np.any(attack_success)
                else math.nan
            ),
            "clean_degradation_rate": (
                float(
                    np.mean(
                        (~clean_topk_correct)[clean_correct]
                    )
                )
                if np.any(clean_correct)
                else math.nan
            ),
        }

    (out_dir / "summary.json").write_text(
        json.dumps(
            summary,
            indent=2,
            allow_nan=True,
        )
    )

    latency_csv = out_dir / "latency_summary.csv"

    with latency_csv.open("w", newline="") as f:
        fields = [
            "stage",
            "n",
            "mean",
            "median",
            "p95",
            "p99",
            "min",
            "max",
            "std",
        ]

        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()

        for stage, stats in summary["latency"].items():
            w.writerow({
                "stage": stage,
                **stats,
            })


# ---------------------------------------------------------------------
# main
# ---------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--config",
        required=True,
    )

    ap.add_argument(
        "--output-dir",
        required=True,
    )

    args = ap.parse_args()

    config_path = Path(args.config).resolve()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = json.loads(config_path.read_text())

    dataset_path = (ROOT / cfg["dataset_path"]).resolve()
    checkpoint_path = (ROOT / cfg["checkpoint_path"]).resolve()
    hef_path = Path(cfg["hef_path"]).resolve()

    for required in (
        dataset_path,
        checkpoint_path,
        hef_path,
    ):
        if not required.exists():
            raise FileNotFoundError(required)

    attack_profiles = []

    for attack_spec in cfg["attacks"]:
        for p in attack_spec["profiles"]:
            attack_profiles.append({
                "attack": attack_spec["name"],
                **p,
            })

    total_base = (
        len(cfg["modulations"])
        * len(cfg["snrs"])
        * len(cfg["sample_indices"])
    )

    total_attack = total_base * len(attack_profiles)
    total_defense = total_attack * len(cfg["topk_values"])

    manifest = {
        "config": cfg,
        "config_path": str(config_path),
        "dataset_sha256": sha256_file(dataset_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "hef_sha256": sha256_file(hef_path),
        "expected_base_rows": total_base,
        "attack_profile_count": len(attack_profiles),
        "expected_attack_rows": total_attack,
        "expected_defense_rows": total_defense,
        "threat_model": (
            "A0 digital classifier-input attack; "
            "adversarial examples generated with differentiable "
            "PyTorch AWN surrogate and evaluated on deployed "
            "quantized Hailo-8 AWN."
        ),
    }

    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2)
    )

    base_csv = out_dir / "base_results.csv"
    attack_csv = out_dir / "attack_results.csv"
    defense_csv = out_dir / "defense_results.csv"

    base_key_fields = [
        "modulation",
        "snr_db",
        "sample_index",
    ]

    attack_key_fields = [
        "modulation",
        "snr_db",
        "sample_index",
        "attack",
        "attack_profile",
    ]

    defense_key_fields = [
        "modulation",
        "snr_db",
        "sample_index",
        "attack",
        "attack_profile",
        "topk",
    ]

    done_base = read_done_keys(
        base_csv,
        base_key_fields,
    )

    done_attack = read_done_keys(
        attack_csv,
        attack_key_fields,
    )

    done_defense = read_done_keys(
        defense_csv,
        defense_key_fields,
    )

    base_fields = [
        "modulation",
        "snr_db",
        "sample_index",
        "label",
        "seed",
        "true_start",
        "true_end",
        "region_count",
        "selected_segment_start",
        "selected_segment_end",
        "detected_region_start",
        "detected_region_end",
        "captured_signal_ratio",
        "start_boundary_error",
        "end_boundary_error",
        "missed_signal_samples",
        "false_occupied_samples",
        "clean_input_sha256",
        "clean_pred",
        "clean_correct",
        "embedding_ms",
        "energy_detection_ms",
        "region_postprocess_ms",
        "segmentation_alignment_ms",
        "awn_preprocess_ms",
        "clean_hailo_inference_ms",
        "hilo_backend",
        "run_status",
    ]

    attack_fields = [
        "modulation",
        "snr_db",
        "sample_index",
        "label",
        "attack",
        "attack_profile",
        "eps_interface_value",
        "attack_params_json",
        "clean_pred",
        "clean_correct",
        "attacked_pred",
        "attacked_correct",
        "attack_success",
        "iq_linf",
        "iq_l2",
        "iq_l1",
        "adversarial_sha256",
        "attack_generation_ms",
        "attacked_hailo_inference_ms",
        "attack_pipeline_ms",
        "attack_backend",
        "attack_status",
        "hilo_backend",
        "run_status",
    ]

    defense_fields = [
        "modulation",
        "snr_db",
        "sample_index",
        "label",
        "attack",
        "attack_profile",
        "topk",
        "clean_correct",
        "attacked_correct",
        "attack_success",
        "clean_topk_pred",
        "clean_topk_correct",
        "clean_degraded",
        "defended_pred",
        "defended_correct",
        "recovered",
        "adversarial_sha256",
        "topk_transform_ms",
        "defended_hailo_inference_ms",
        "defense_pipeline_ms",
        "topk_backend",
        "topk_status",
        "hilo_backend",
        "run_status",
    ]

    print(
        f"[plan] base={total_base} "
        f"attack_profiles={len(attack_profiles)} "
        f"attack_rows={total_attack} "
        f"defense_rows={total_defense}"
    )

    t_dataset = time.perf_counter()
    dataset = load_radioml_dict(str(dataset_path))
    dataset_load_ms = elapsed_ms(t_dataset)

    manifest["dataset_load_ms"] = dataset_load_ms

    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2)
    )

    cpu = AWNModelAdapter(
        checkpoint_path=str(checkpoint_path),
        device="cpu",
    )

    if (
        cpu.model is None
        or cpu.status != "ok"
        or "dummy" in cpu.backend_name.lower()
    ):
        raise RuntimeError(
            "PyTorch AWN unavailable or fallback: "
            f"{cpu.backend_name}, status={cpu.status}"
        )

    cpu.model.eval()

    attack_adapter = AttackAdapter(
        awn_model=cpu.model,
        device="cpu",
    )

    topk_adapter = TopKAdapter()

    start_all = time.perf_counter()

    completed_attack_calls = 0

    with HailoAWNAdapter(str(hef_path)) as hailo:

        for mod in cfg["modulations"]:
            label = int(RML2016_10A_CLASSES[mod])

            for snr in cfg["snrs"]:
                key = (mod, int(snr))

                if key not in dataset:
                    raise RuntimeError(
                        f"dataset missing {key}"
                    )

                block = dataset[key]

                for sample_index in cfg["sample_indices"]:

                    base_key = (
                        str(mod),
                        str(snr),
                        str(sample_index),
                    )

                    all_this_base_done = True

                    for prof in attack_profiles:
                        akey = (
                            str(mod),
                            str(snr),
                            str(sample_index),
                            str(prof["attack"]),
                            str(prof["id"]),
                        )

                        if akey not in done_attack:
                            all_this_base_done = False
                            break

                        for k in cfg["topk_values"]:
                            dkey = akey + (str(k),)

                            if dkey not in done_defense:
                                all_this_base_done = False
                                break

                    if all_this_base_done:
                        continue

                    if sample_index >= block.shape[0]:
                        raise RuntimeError(
                            f"{key} has only "
                            f"{block.shape[0]} samples"
                        )

                    sample = block[sample_index].astype(
                        np.float32
                    )

                    if sample.shape != (2, 128):
                        raise RuntimeError(
                            f"bad sample shape {sample.shape}"
                        )

                    x_clean, sense_meta, sense_time = (
                        build_sensing_input(
                            sample,
                            cfg,
                            mod,
                            int(snr),
                            int(sample_index),
                        )
                    )

                    t0 = time.perf_counter()
                    clean_logits, clean_meta = hailo.infer(
                        x_clean
                    )
                    clean_hailo_ms = elapsed_ms(t0)

                    if is_bad_meta(clean_meta, "awn"):
                        raise RuntimeError(
                            f"Hailo clean fallback/error: "
                            f"{clean_meta}"
                        )

                    clean_pred = int(
                        np.argmax(clean_logits[0])
                    )
                    clean_correct = (
                        clean_pred == label
                    )

                    if base_key not in done_base:
                        base_row = {
                            "modulation": mod,
                            "snr_db": snr,
                            "sample_index": sample_index,
                            "label": label,
                            **sense_meta,
                            "clean_pred": clean_pred,
                            "clean_correct": clean_correct,
                            **sense_time,
                            "clean_hailo_inference_ms":
                                clean_hailo_ms,
                            "hilo_backend":
                                clean_meta["awn_backend"],
                            "run_status": "ok",
                        }

                        csv_append(
                            base_csv,
                            base_row,
                            base_fields,
                        )

                        done_base.add(base_key)

                    clean_topk_cache = {}

                    for k in cfg["topk_values"]:
                        t0 = time.perf_counter()

                        x_clean_k, tk_meta = (
                            topk_adapter.apply(
                                x_clean,
                                topk=int(k),
                            )
                        )

                        clean_topk_transform_ms = (
                            elapsed_ms(t0)
                        )

                        if is_bad_meta(tk_meta, "topk"):
                            raise RuntimeError(
                                "Top-K clean fallback/error: "
                                f"{tk_meta}"
                            )

                        t0 = time.perf_counter()

                        clean_k_logits, clean_k_meta = (
                            hailo.infer(x_clean_k)
                        )

                        clean_k_infer_ms = elapsed_ms(t0)

                        if is_bad_meta(
                            clean_k_meta,
                            "awn",
                        ):
                            raise RuntimeError(
                                "Hailo clean Top-K "
                                f"fallback/error: "
                                f"{clean_k_meta}"
                            )

                        clean_k_pred = int(
                            np.argmax(clean_k_logits[0])
                        )

                        clean_topk_cache[int(k)] = {
                            "pred": clean_k_pred,
                            "correct": (
                                clean_k_pred == label
                            ),
                            "transform_ms":
                                clean_topk_transform_ms,
                            "infer_ms":
                                clean_k_infer_ms,
                        }

                    for prof in attack_profiles:
                        attack = prof["attack"]
                        profile_id = prof["id"]

                        akey = (
                            str(mod),
                            str(snr),
                            str(sample_index),
                            str(attack),
                            str(profile_id),
                        )

                        defense_missing = any(
                            akey + (str(k),)
                            not in done_defense
                            for k in cfg["topk_values"]
                        )

                        if (
                            akey in done_attack
                            and not defense_missing
                        ):
                            continue

                        eps = float(
                            prof.get("eps", 0.03)
                        )

                        params = dict(
                            prof.get("params", {})
                        )

                        cw_c = float(
                            prof.get("cw_c", 1.0)
                        )
                        cw_steps = int(
                            prof.get("cw_steps", 20)
                        )
                        cw_lr = float(
                            prof.get("cw_lr", 0.01)
                        )

                        attack_seed = derive_seed(
                            int(
                                cfg["sensing"][
                                    "base_seed"
                                ]
                            ),
                            mod,
                            int(snr),
                            int(sample_index),
                            f"{attack}|{profile_id}",
                        )

                        cpu.model.eval()

                        t_attack_pipeline = (
                            time.perf_counter()
                        )

                        t0 = time.perf_counter()

                        x_adv, attack_meta = (
                            attack_adapter.apply(
                                x_clean,
                                attack=attack,
                                eps=eps,
                                temperature=1.0,
                                seed=attack_seed,
                                diagnostics=False,
                                cw_c=cw_c,
                                cw_steps=cw_steps,
                                cw_lr=cw_lr,
                                attack_params=params,
                            )
                        )

                        attack_ms = elapsed_ms(t0)

                        if cpu.model.training:
                            raise RuntimeError(
                                "AttackAdapter left PyTorch "
                                "AWN in train mode"
                            )

                        if is_bad_meta(
                            attack_meta,
                            "attack",
                        ):
                            raise RuntimeError(
                                "Attack fallback/error: "
                                f"{attack_meta}"
                            )

                        if x_adv.shape != x_clean.shape:
                            raise RuntimeError(
                                "attack shape mismatch "
                                f"{x_clean.shape} -> "
                                f"{x_adv.shape}"
                            )

                        if not np.isfinite(x_adv).all():
                            raise RuntimeError(
                                "adversarial IQ "
                                "contains NaN/Inf"
                            )

                        adv_hash = sha256_array(x_adv)

                        t0 = time.perf_counter()

                        attacked_logits, attacked_meta = (
                            hailo.infer(x_adv)
                        )

                        attacked_hailo_ms = (
                            elapsed_ms(t0)
                        )

                        if is_bad_meta(
                            attacked_meta,
                            "awn",
                        ):
                            raise RuntimeError(
                                "Hailo attacked "
                                f"fallback/error: "
                                f"{attacked_meta}"
                            )

                        attacked_pred = int(
                            np.argmax(
                                attacked_logits[0]
                            )
                        )

                        attacked_correct = (
                            attacked_pred == label
                        )

                        attack_success = (
                            clean_correct
                            and not attacked_correct
                        )

                        delta = x_adv - x_clean

                        iq_linf = float(
                            np.max(np.abs(delta))
                        )
                        iq_l2 = float(
                            np.linalg.norm(
                                delta.reshape(-1)
                            )
                        )
                        iq_l1 = float(
                            np.sum(np.abs(delta))
                        )

                        attack_pipeline_ms = (
                            elapsed_ms(
                                t_attack_pipeline
                            )
                        )

                        if akey not in done_attack:
                            attack_row = {
                                "modulation": mod,
                                "snr_db": snr,
                                "sample_index":
                                    sample_index,
                                "label": label,
                                "attack": attack,
                                "attack_profile":
                                    profile_id,
                                "eps_interface_value":
                                    eps,
                                "attack_params_json":
                                    json.dumps(
                                        params,
                                        sort_keys=True,
                                    ),
                                "clean_pred":
                                    clean_pred,
                                "clean_correct":
                                    clean_correct,
                                "attacked_pred":
                                    attacked_pred,
                                "attacked_correct":
                                    attacked_correct,
                                "attack_success":
                                    attack_success,
                                "iq_linf": iq_linf,
                                "iq_l2": iq_l2,
                                "iq_l1": iq_l1,
                                "adversarial_sha256":
                                    adv_hash,
                                "attack_generation_ms":
                                    attack_ms,
                                "attacked_hailo_inference_ms":
                                    attacked_hailo_ms,
                                "attack_pipeline_ms":
                                    attack_pipeline_ms,
                                "attack_backend":
                                    attack_meta[
                                        "attack_backend"
                                    ],
                                "attack_status":
                                    attack_meta[
                                        "attack_status"
                                    ],
                                "hilo_backend":
                                    attacked_meta[
                                        "awn_backend"
                                    ],
                                "run_status": "ok",
                            }

                            csv_append(
                                attack_csv,
                                attack_row,
                                attack_fields,
                            )

                            done_attack.add(akey)

                        for k in cfg["topk_values"]:
                            dkey = akey + (str(k),)

                            if dkey in done_defense:
                                continue

                            t_def_pipeline = (
                                time.perf_counter()
                            )

                            t0 = time.perf_counter()

                            x_def, tk_meta = (
                                topk_adapter.apply(
                                    x_adv,
                                    topk=int(k),
                                )
                            )

                            topk_ms = elapsed_ms(t0)

                            if is_bad_meta(
                                tk_meta,
                                "topk",
                            ):
                                raise RuntimeError(
                                    "Top-K attacked "
                                    f"fallback/error: "
                                    f"{tk_meta}"
                                )

                            t0 = time.perf_counter()

                            def_logits, def_meta = (
                                hailo.infer(x_def)
                            )

                            defended_hailo_ms = (
                                elapsed_ms(t0)
                            )

                            if is_bad_meta(
                                def_meta,
                                "awn",
                            ):
                                raise RuntimeError(
                                    "Hailo defended "
                                    f"fallback/error: "
                                    f"{def_meta}"
                                )

                            defended_pred = int(
                                np.argmax(
                                    def_logits[0]
                                )
                            )

                            defended_correct = (
                                defended_pred
                                == label
                            )

                            recovered = (
                                attack_success
                                and defended_correct
                            )

                            ctk = clean_topk_cache[
                                int(k)
                            ]

                            clean_degraded = (
                                clean_correct
                                and not ctk["correct"]
                            )

                            defense_pipeline_ms = (
                                elapsed_ms(
                                    t_def_pipeline
                                )
                            )

                            defense_row = {
                                "modulation": mod,
                                "snr_db": snr,
                                "sample_index":
                                    sample_index,
                                "label": label,
                                "attack": attack,
                                "attack_profile":
                                    profile_id,
                                "topk": k,
                                "clean_correct":
                                    clean_correct,
                                "attacked_correct":
                                    attacked_correct,
                                "attack_success":
                                    attack_success,
                                "clean_topk_pred":
                                    ctk["pred"],
                                "clean_topk_correct":
                                    ctk["correct"],
                                "clean_degraded":
                                    clean_degraded,
                                "defended_pred":
                                    defended_pred,
                                "defended_correct":
                                    defended_correct,
                                "recovered": recovered,
                                "adversarial_sha256":
                                    adv_hash,
                                "topk_transform_ms":
                                    topk_ms,
                                "defended_hailo_inference_ms":
                                    defended_hailo_ms,
                                "defense_pipeline_ms":
                                    defense_pipeline_ms,
                                "topk_backend":
                                    tk_meta[
                                        "topk_backend"
                                    ],
                                "topk_status":
                                    tk_meta[
                                        "topk_status"
                                    ],
                                "hilo_backend":
                                    def_meta[
                                        "awn_backend"
                                    ],
                                "run_status": "ok",
                            }

                            csv_append(
                                defense_csv,
                                defense_row,
                                defense_fields,
                            )

                            done_defense.add(dkey)

                        completed_attack_calls += 1

                        if (
                            completed_attack_calls % 25 == 0
                        ):
                            elapsed = (
                                time.perf_counter()
                                - start_all
                            )

                            print(
                                "[progress] "
                                f"attack_rows="
                                f"{len(done_attack)}/"
                                f"{total_attack} "
                                f"defense_rows="
                                f"{len(done_defense)}/"
                                f"{total_defense} "
                                f"elapsed="
                                f"{elapsed:.1f}s",
                                flush=True,
                            )

    if len(done_base) != total_base:
        raise RuntimeError(
            f"base result count mismatch: "
            f"{len(done_base)} != {total_base}"
        )

    if len(done_attack) != total_attack:
        raise RuntimeError(
            f"attack result count mismatch: "
            f"{len(done_attack)} != "
            f"{total_attack}"
        )

    if len(done_defense) != total_defense:
        raise RuntimeError(
            f"defense result count mismatch: "
            f"{len(done_defense)} != "
            f"{total_defense}"
        )

    build_summary(
        out_dir,
        attack_csv,
        defense_csv,
        base_csv,
    )

    manifest["completed_base_rows"] = len(
        done_base
    )
    manifest["completed_attack_rows"] = len(
        done_attack
    )
    manifest["completed_defense_rows"] = len(
        done_defense
    )
    manifest["run_status"] = "complete"

    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2)
    )

    print()
    print("=== FULL MATRIX COMPLETE ===")
    print(f"base rows    : {len(done_base)}")
    print(f"attack rows  : {len(done_attack)}")
    print(f"defense rows : {len(done_defense)}")
    print(f"output       : {out_dir}")


if __name__ == "__main__":
    main()
