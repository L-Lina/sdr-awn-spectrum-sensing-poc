"""
cProfile + torch.profiler breakdown of FGSM/PGD/CW attack generation, using
the exact real AttackAdapter/AWNModelAdapter path every other formal script
in this repo uses. Profiles a modest, explicitly-recorded number of calls
(not the full 330-sample baseline -- profiling overhead itself distorts
wall-clock time, so this is a separate, smaller run whose OWN sample count
is recorded in its output, never conflated with Phase B's baseline numbers).
"""

from __future__ import annotations

import cProfile
import io
import pstats
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.adapters.attack_adapter import AttackAdapter, _REAL_ATTACK_SOURCE  # noqa: E402
from src.adapters.awn_adapter import AWNModelAdapter, _REAL_MODEL_SOURCE  # noqa: E402
from src.sensing.energy_detection import energy_detect, filter_by_min_length, mask_to_regions, merge_close_regions  # noqa: E402
from src.sensing.normalize import apply_awn_preprocess, to_awn_input  # noqa: E402
from src.sensing.radioml_source import embed_sample_in_noise, load_radioml_sample  # noqa: E402
from src.sensing.segmentation import select_aligned_segments  # noqa: E402

DATASET_PATH = "/home/xiaomi/adversarial-rf/data/RML2016.10a_dict.pkl"
CHECKPOINT_PATH = "external/adversarial-rf/2016.10a_AWN.pkl"
N_PROFILE_CALLS = 30
N_WARMUP = 10


def build_clean_input(mod: str, snr: int, idx: int) -> np.ndarray:
    sample = load_radioml_sample(DATASET_PATH, mod, snr, idx)
    iq, _ = embed_sample_in_noise(sample, 8192, 20.0, seed=idx)
    mask = energy_detect(iq, window=128, threshold_factor=5.0)
    regions = filter_by_min_length(merge_close_regions(mask_to_regions(mask), merge_gap=0), min_len=128)
    segments, _ = select_aligned_segments(iq, regions, seg_len=128, policy="max-energy", hop=1)
    x = apply_awn_preprocess(segments[:1], policy="radioml-native")
    return to_awn_input(x, seg_len=128)


def main() -> None:
    out_dir = Path(sys.argv[sys.argv.index("--output-dir") + 1])
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[profile] loading real AWN + attack backends ...", flush=True)
    awn = AWNModelAdapter(checkpoint_path=CHECKPOINT_PATH, device="cpu")
    assert awn.backend_name == _REAL_MODEL_SOURCE and awn.status == "ok"
    attack = AttackAdapter(awn_model=awn.model, device="cpu")
    assert attack.wrapped_model is not None and attack.backend_name == _REAL_ATTACK_SOURCE
    print("[profile] real backends confirmed", flush=True)

    x_inputs = [build_clean_input("QPSK", 0, i % 10) for i in range(N_WARMUP + N_PROFILE_CALLS)]

    for attack_name, eps, params in [("fgsm", 0.05, {"eps": 0.05}), ("pgd", 0.05, {"eps": 0.05}), ("cw", 0.05, {})]:
        print(f"[profile] warm-up {attack_name}: {N_WARMUP} calls", flush=True)
        for i in range(N_WARMUP):
            attack.apply(x_inputs[i], attack=attack_name, eps=eps, seed=0, attack_params=params)

        print(f"[profile] cProfile {attack_name}: {N_PROFILE_CALLS} calls", flush=True)
        profiler = cProfile.Profile()
        profiler.enable()
        for i in range(N_PROFILE_CALLS):
            attack.apply(x_inputs[N_WARMUP + i], attack=attack_name, eps=eps, seed=0, attack_params=params)
        profiler.disable()

        buf = io.StringIO()
        ps = pstats.Stats(profiler, stream=buf).sort_stats("cumulative")
        ps.print_stats(40)
        (out_dir / f"{attack_name}_cprofile_top40.txt").write_text(
            f"# cProfile output for {N_PROFILE_CALLS} calls to AttackAdapter.apply(attack='{attack_name}'), "
            f"after {N_WARMUP} warm-up calls (excluded). Sorted by cumulative time.\n\n" + buf.getvalue()
        )

        buf2 = io.StringIO()
        ps2 = pstats.Stats(profiler, stream=buf2).sort_stats("tottime")
        ps2.print_stats(40)
        (out_dir / f"{attack_name}_cprofile_top40_selftime.txt").write_text(
            f"# cProfile output for {N_PROFILE_CALLS} calls to AttackAdapter.apply(attack='{attack_name}'), "
            f"after {N_WARMUP} warm-up calls (excluded). Sorted by self (tottime).\n\n" + buf2.getvalue()
        )
        print(f"[profile] {attack_name} cProfile written", flush=True)

        # torch.profiler pass, separate from cProfile pass (different instrumentation overhead)
        try:
            import torch
            from torch.profiler import ProfilerActivity, profile

            with profile(activities=[ProfilerActivity.CPU], record_shapes=False) as prof:
                for i in range(N_PROFILE_CALLS):
                    attack.apply(x_inputs[N_WARMUP + i], attack=attack_name, eps=eps, seed=0, attack_params=params)
            table = prof.key_averages().table(sort_by="cpu_time_total", row_limit=30)
            (out_dir / f"{attack_name}_torch_profiler_top30.txt").write_text(
                f"# torch.profiler CPU breakdown for {N_PROFILE_CALLS} calls to AttackAdapter.apply(attack='{attack_name}')\n\n"
                + table
            )
            print(f"[profile] {attack_name} torch.profiler written", flush=True)
        except Exception as exc:  # noqa: BLE001 -- profiler tooling failure must not silently corrupt the cProfile results already written
            (out_dir / f"{attack_name}_torch_profiler_FAILED.txt").write_text(
                f"torch.profiler pass failed: {type(exc).__name__}: {exc}\n"
            )
            print(f"[profile] {attack_name} torch.profiler FAILED: {exc}", flush=True)

    print("[profile] DONE", flush=True)


if __name__ == "__main__":
    main()
