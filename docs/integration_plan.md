# Integration Plan: SDR sensing -> AWN -> Attack -> Top-K Defender

Status: inspection complete, no new code written yet. This document maps what
already exists in the two vendored repos (`external/AWN`, `external/adversarial-rf`)
to what this glue repo needs to build, and proposes a minimal first experiment.

## 0. Repo relationship

`external/adversarial-rf` (nigelzzz) is a superset/fork-in-spirit of `external/AWN`
(zjwfufu): it vendors its own copy of `AWN_All.py` / `models/model.py` /
`models/lifting.py` (near-identical to upstream AWN) and adds the entire
attack/defense/experiment layer on top. In other words:

- **Model + data loading contract** -> reuse from either; `adversarial-rf`'s copy
  is the one that's actually wired to attacks/defenses, so treat it as canonical.
- **Attack implementations** -> only in `adversarial-rf` (`util/adv_attack.py`,
  `util/adaptive_attack.py`, torchattacks integration).
- **Top-K / adaptive-K defender** -> only in `adversarial-rf` (`util/defense.py`,
  `util/adaptive_defense.py`, `util/adaptive_k_calibration.py`, and the
  `adaptive_k_*.py` experiment scripts at repo root, matching
  `reports/adaptive_k_report_CN.md`).

This means the front-end pipeline in this repo (`scripts/sdr_sensing_to_awn_poc.py`)
should ultimately talk to **`external/adversarial-rf`**'s model/attack/defense code,
not to plain upstream AWN. `external/AWN` is kept as a submodule mainly as the
"clean reference" for the model architecture and paper-faithful training/eval flow.

## 1. What can be reused directly (no adapter needed)

| Need | Reuse from | Notes |
|---|---|---|
| AWN model class | `external/adversarial-rf/models/model.py:AWN` | `forward(x)` where `x: [N, 2, T]` (T=128 for RML2016) returns `(logit, regu_sum)` — **tuple, not just logits** |
| Pretrained checkpoints | `external/adversarial-rf/2016.10a_AWN.pkl`, `2016.10b_AWN.pkl`, `2018.01a_AWN.pkl` (repo root, already committed — not gitignored) | Load with `torch.load(path, map_location=device)` then `model.load_state_dict(...)`; no need to source checkpoints separately |
| Model construction | `external/adversarial-rf/util/utils.py:create_AWN_model(cfg)` | Needs a `cfg` with `num_classes, num_level, in_channels, kernel_size, latent_dim, regu_details, regu_approx` — see `config/2016.10a.yml` for values (`num_classes=11, in_channels=64, kernel_size=3, latent_dim=320, num_level=1, regu_details=0.01, regu_approx=0.01`) |
| Top-K FFT denoise (defender core) | `util/defense.py:fft_topk_denoise(x: [N,2,T], topk: int) -> Tensor` | Exactly matches our pipeline's `[N, 2, 128]` tensor shape — directly callable on `to_awn_input()` output, no reshaping |
| Adaptive-K defense (no detector) | `util/defense.py:adaptive_k_defense(x, ratio_thresh=0.05)` and `adaptive_k_v2_defense(x, ratio_thresh, k_max, flatness_threshold, quant_levels)` | Pure signal-processing, per-sample K selection via spectral knee — this is the "Top-K defender" architecture referenced in `reports/adaptive_k_report_CN.md` |
| Detector-gated Top-K | `util/detector.py:detector_gate_fft_topk(x, detector, threshold, topk)` | Needs a trained `RFSignalAutoEncoder` checkpoint (`checkpoint/detector_ae.pth`) — optional, more advanced path |
| Attack <-> torchattacks bridge | `util/adv_attack.py:Model01Wrapper`, `iq_to_ta_input`/`ta_output_to_iq` (+ `_minmax` variants) | Converts our `[-1,1]` IQ range to the `[0,1]` 4D image-style tensors torchattacks expects, and unwraps the model's `(logit, regu)` tuple to bare logits |
| Internal CW attack | `util/adv_attack.py:cw_l2_attack(...)` | Available without torchattacks if we want to avoid that dependency initially |
| Reference CLI patterns | `external/adversarial-rf/main.py` (`--mode adv_eval`, `multi_attack_eval`, `sigguard_eval`, `calibrate_adaptive_k`) | Not reused as code, but as the spec for argument names/semantics we should mirror in our own glue scripts |

## 2. What needs an adapter

None of the reusable pieces above require **rewriting**, but a few need a thin
adapter layer because they assume a fuller experiment context (a `cfg` object,
a `logger`, on-disk directory conventions) that our lightweight PoC pipeline
doesn't have:

1. **`AwnModelAdapter`** (new, thin) — wraps `create_AWN_model(cfg)` +
   checkpoint loading behind a single call:
   `load_awn(checkpoint_path, dataset="2016.10a") -> nn.Module` that builds the
   minimal config internally (from the same values as `config/2016.10a.yml`)
   instead of requiring a full `Config` object with YAML/log-dir side effects.
   TODO: confirm the vendored `adversarial-rf` config values are identical to
   upstream AWN's before hardcoding them (spot-checked `2016.10a.yml`, matches).

2. **`run_awn_inference(x)` real implementation** — replace the current numpy
   placeholder in `scripts/sdr_sensing_to_awn_poc.py` with a torch path that:
   - loads the model once (via `AwnModelAdapter`, cached/lazy)
   - calls `logit, _ = model(x)` (must unpack the tuple — this is the #1 footgun
     when wiring in the real model)
   - returns `logit` only, keeping the rest of the pipeline (defense hooks,
     reporting) untouched.
   TODO mark this explicitly in code until the swap is done.

3. **Front-end -> attack/defense boundary adapter** — our sensing pipeline
   produces normalized-per-segment `[N, 2, 128]` float32. `adversarial-rf`'s
   attack/defense code assumes IQ in a specific range convention (`[-1, 1]`,
   or "typical amplitude ~±0.02" per its own docs) depending on which function
   is used. Our per-segment unit-average-power normalization does **not**
   automatically match either convention. TODO: adapter function
   `to_attack_domain(x)` / `from_attack_domain(x)` to rescale between our
   sensing-side normalization and whatever the attack/defense functions expect
   — needs to be nailed down empirically once we run real inference (see
   open question in Section 4).

4. **No adapter needed, but a naming decision**: our PoC's `SEGMENT_LEN=128`
   and `--window-size` already match RML2016's `T=128`. If we ever point this
   pipeline at RML2018.01a-trained models (`T=1024`), the whole front-end
   (`window-size`, energy-detection window) needs to change together — not
   just the model checkpoint.

## 3. Where the spectrum-sensing front-end plugs in

Current front-end (`scripts/sdr_sensing_to_awn_poc.py`, already built and tested):

```
IQ stream -> energy_detect -> extract_occupied_regions -> segment_regions
          -> normalize_segments -> to_awn_input()  =>  [N, 2, 128] float32
```

Plug points, in order:

1. **`to_awn_input()` output** is the hand-off point to the model. This is
   already shaped correctly for `AWN.forward()` — just needs `torch.from_numpy(x)`.
2. **Attack insertion point**: between `to_awn_input()` and `run_awn_inference()`.
   Attacks need gradients through the model, so this step requires the real
   torch model loaded (not the placeholder) plus `Model01Wrapper` if using
   torchattacks, or `cw_l2_attack` directly for the internal CW path.
3. **Defense insertion point**: also between `to_awn_input()` and
   `run_awn_inference()` (or, in the detector-gated case, wrapping the
   inference call itself). `fft_topk_denoise` / `adaptive_k_defense` /
   `adaptive_k_v2_defense` all take and return `[N, 2, T]` tensors, so they drop
   in immediately after `to_awn_input()` (attack case) or in place of it
   (no-attack, defense-only sanity check case).
4. **What does NOT change**: `energy_detect` / `segment_regions` /
   `normalize_segments` are sensing-side only and have no equivalent in
   `adversarial-rf` (that repo starts from pre-segmented RML dataset samples,
   never from a raw stream) — this front-end is this glue repo's actual
   original contribution, not something to reconcile with upstream.

## 4. Minimal first experiment (proposed, not yet implemented)

Goal: prove the full chain end-to-end on **synthetic sensing output**, using
the **real** AWN model and **real** Top-K defense, with the attack step
optional/toggle-able. Deliberately small in scope:

1. Extend `run_awn_inference(x)` to optionally load the real model
   (behind a flag, e.g. `--real-model checkpoint.pkl`) instead of always using
   the placeholder — keep the placeholder as the default so `--demo` stays
   dependency-free.
2. Run the existing synthetic `--demo` pipeline to get real `[N, 2, 128]`
   segments (already working, from the previous PoC step).
3. Feed those segments through the real AWN model (`2016.10a_AWN.pkl`) and
   print predicted modulation labels instead of random logits.
4. Apply `fft_topk_denoise(x, topk=50)` (matches `adversarial-rf`'s own default
   `--sigguard_topk 50`) both with and without an attack step, and compare
   predictions before/after — this alone validates the defender wiring without
   needing torchattacks yet.
5. **Stretch**: add one attack (FGSM via `Model01Wrapper`, since it's the
   cheapest attack to get working) to see the defended-vs-undefended accuracy
   gap on synthetic-burst-derived segments, mirroring `sigguard_eval`'s output
   table format.

Explicitly out of scope for this first experiment (per current instructions):
no training, no real RML dataset download, no dependency installation — steps
2-5 above are a design target for the *next* implementation phase, not
something to run yet.

## 5. Open questions / risks

- **Normalization mismatch (see adapter #3)**: real RML2016 signals have
  amplitude ~±0.02 with the model trained on that scale; our sensing pipeline's
  unit-average-power normalization will NOT numerically match that unless we
  either (a) match `adversarial-rf`'s normalization convention exactly, or
  (b) verify empirically that the trained model is scale-invariant enough
  in practice. Needs a real run to check classification accuracy on known
  synthetic bursts before trusting any prediction.
- **`external/adversarial-rf` is a large, actively-evolving research repo**
  (100+ scripts, its own `CLAUDE.md`/`AGENTS.md` workflow conventions, a `.planning/`
  GSD workflow). We are only *reading* specific files from it as a pinned
  submodule commit — we should not edit inside it, and should re-pin
  (`git submodule update --remote`) deliberately, not accidentally.
- **Two AWN copies drift risk**: `external/AWN` and
  `external/adversarial-rf/models/model.py` are similar but not
  guaranteed byte-identical. Section 0 already decided to standardize on
  `adversarial-rf`'s copy for anything touching attack/defense; `external/AWN`
  is reference-only. Worth a one-time diff check before real integration work
  starts.
- **Dependencies**: real inference requires `torch` (and `torchattacks` for
  the attack bridge), none of which are installed yet per current constraints.

## 6. Summary: next implementation phase (not started)

1. Add `AwnModelAdapter` (adapter #1) as new code in this repo (not editing
   the submodules).
2. Swap `run_awn_inference` placeholder for the real-model path behind a flag
   (adapter #2), keep placeholder as default.
3. Resolve the normalization question (Section 5) empirically.
4. Wire in `fft_topk_denoise` as the first defense integration (simplest,
   already shape-compatible, no detector checkpoint needed).
5. Only after 1-4 work end-to-end: add attack step via `Model01Wrapper`.
