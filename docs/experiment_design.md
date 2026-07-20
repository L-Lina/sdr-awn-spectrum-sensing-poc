# Experiment Design (Phase 1: skeleton)

Status: project structure + dry-run runner skeleton. No real AWN model, no
real attack library, no real Top-K defender wired in yet -- everything below
the sensing front-end is a numpy-only placeholder. See
`docs/integration_plan.md` for how each placeholder maps to the real
implementation in `external/adversarial-rf`.

## Project layout

```
experiments/
  run_full_experiment.py   single-run CLI entrypoint
  run_batch.py              parameter-grid sweep over the same pipeline
src/
  sensing/                  IQ generation, energy detection, windowing, normalization
    iq_source.py            synthetic IQ generator + .cfile reader (for later)
    radioml_source.py        RML2016.10a real-sample loader + single/multi-burst
                              noise-stream embedding (--iq-source radioml, --num-bursts;
                              see docs/parameter_validation.md sections 14-15)
    ground_truth_metrics.py  single- and multi-burst detection/boundary/capture-ratio/
                              Pd-Pfa metrics against known true burst interval(s)
                              (radioml mode only, currently)
    energy_detection.py      energy_detect / mask_to_regions / merge_close_regions / filter_by_min_length
    segmentation.py           segment_regions
    normalize.py               normalize_segments / to_awn_input
  adapters/                 placeholder model/attack/defense adapters
    awn_adapter.py            dummy_awn_inference (numpy random logits)
    attack_adapter.py         dummy_attack (deterministic sign-noise perturbation)
    defense_adapter.py        dummy_topk_defense (real FFT Top-K, numpy-only)
  utils/                    shared plumbing used by both experiment entrypoints
    config.py                 ExperimentConfig dataclass + shared argparse
    pipeline.py                run_dry_run_experiment() -- the actual orchestration
    csv_writer.py               summary.csv writer (stdlib csv, no pandas)
    plotting.py                  sensing plot (matplotlib, best-effort/optional)
configs/                    reserved for future YAML/JSON experiment configs (empty for now)
results/                    run outputs land here (gitignored except .gitkeep)
docs/
  integration_plan.md       AWN / adversarial-rf reuse-vs-adapter mapping
  experiment_design.md      this file
scripts/sdr_sensing_to_awn_poc.py   original standalone PoC (untouched, kept as-is)
```

`scripts/sdr_sensing_to_awn_poc.py` is left as it was -- the new `src/sensing/`
package is a re-organized, parameterized version of the same logic (plus
`merge_gap` / `min_region_len` support it didn't have), not a replacement edit
of that file.

## CLI parameters (`run_full_experiment.py`)

| Flag | Meaning | Default |
|---|---|---|
| `--snr` | Synthetic burst SNR in dB (drives generated noise level) | `10.0` |
| `--mod` | Modulation label tag; cosmetic only in this phase (varies burst freq offset, does not synthesize a real modulation waveform) | `BPSK` |
| `--attack` | Attack name placeholder (`none`, `fgsm`, `pgd`, ...) -- only affects placeholder logging/perturbation magnitude, not a real attack yet | `none` |
| `--topk` | Number of FFT bins kept by the Top-K defense placeholder | `50` |
| `--threshold-factor` | Energy detection threshold = median windowed power * this factor | `5.0` |
| `--window-size` | Segment length / energy-detection window (AWN expects 128) | `128` |
| `--min-region-len` | Minimum occupied region length to keep; defaults to `--window-size` if unset | `None` |
| `--merge-gap` | Merge occupied regions separated by <= this many samples before length filtering | `0` |
| `--burst-len` | Synthetic burst length in samples | `600` |
| `--output-dir` | Where `summary.csv` and `sensing_plot.png` are written | `results/run` |
| `--dry-run` | Required in this phase -- selects the placeholder pipeline | `False` |

`run_batch.py` exposes the same tail flags plus `--snr-list` / `--mod-list` /
`--attack-list` / `--topk-list` (comma-separated) to sweep a grid, writing one
subdirectory per combination under `--output-dir` plus an aggregated
`batch_summary.csv`.

This table (and the pipeline diagram below) describe this doc's original
Phase-1 dummy-only skeleton and have not been comprehensively refreshed since
-- most placeholders it describes are now wired to real backends (see
`docs/parameter_validation.md`, which is the actively-maintained, currently
accurate parameter/status record; this file is kept only for the original
project-layout overview and is not the source of truth for current
behavior). One addition directly relevant to project layout: `--iq-source
{synthetic,radioml}` (default `synthetic`) selects between the synthetic
generator above and a real RML2016.10a sample embedded in a synthetic noise
stream (`--dataset-path`/`--dataset-mod`/`--dataset-snr`/`--sample-index`,
all required in `radioml` mode) -- see `docs/parameter_validation.md`
section 14 for the full data flow, ground-truth metrics, ground-truth
metric formulas, and functional-test results.

## Dry-run pipeline

```
generate_synthetic_iq(snr, mod, burst_len)
  -> validate_iq
  -> energy_detect(window=window_size, threshold_factor)
  -> mask_to_regions -> merge_close_regions(merge_gap) -> filter_by_min_length(min_region_len)
  -> segment_regions(seg_len=window_size)
  -> normalize_segments
  -> to_awn_input()                          => x_clean  [N, 2, window_size] float32
  -> dummy_awn_inference(x_clean)             => logits_clean
  -> dummy_attack(x_clean, attack)            => x_adv
  -> dummy_awn_inference(x_adv)               => logits_attacked
  -> dummy_topk_defense(x_adv, topk)          => x_defended
  -> dummy_awn_inference(x_defended)          => logits_defended
  -> summary.csv (per-segment predictions)
  -> sensing_plot.png (best-effort; skipped if matplotlib isn't installed)
```

## What's real vs placeholder right now

| Stage | Status |
|---|---|
| Synthetic IQ generation, energy detection, windowing, normalization | Real, numpy-only |
| `dummy_topk_defense` | Real FFT Top-K algorithm (numpy), same math as the real defender -- just not the actual `external/adversarial-rf` code path yet |
| `dummy_awn_inference` | Fully placeholder -- random logits, no model |
| `dummy_attack` | Fully placeholder -- fixed-magnitude sign-noise, not gradient-based |

## Next phases (not started)

1. Wire `awn_adapter.py` to the real AWN model + checkpoint (`external/adversarial-rf`).
2. Wire `attack_adapter.py` to `Model01Wrapper` + torchattacks (or internal CW).
3. Wire `defense_adapter.py` to the real `fft_topk_denoise` / `adaptive_k_defense`.
4. Populate `configs/` with saved parameter sets once the grid shape stabilizes.

Each of these requires `torch` (and `torchattacks` for phase 2), which is not
installed yet by design -- this phase only exercises the skeleton.
