# Hermes Project Instructions

## Project mission

This repository is an offline proof of concept for connecting SDR/GNU Radio complex-IQ data to energy-triggered Automatic Modulation Classification (AMC), adversarial evaluation, and Top-K spectral preprocessing around AWN.

Describe the implemented system precisely as:

> Energy-triggered AMC on a pre-channelized complex-baseband stream.

The present code is not a wideband spectrum sensor, an operational communications receiver, or an autonomous control-plane system. It does not yet provide a live UHD/ZMQ stream, a channelizer, demodulation/decoding, CRC verification, protocol parsing, or autonomous channel allocation.

## Repository map

- `src/sensing/`: IQ sources, temporal energy detection, occupied-region handling, segmentation, normalization, and sensing metrics.
- `src/adapters/`: AWN, attack, and Top-K/defense integration boundaries.
- `src/utils/`: experiment configuration, orchestration, CSV output, plotting, and aggregation.
- `experiments/`: formal single-run, batch, revalidation, and ablation entry points.
- `scripts/sdr_sensing_to_awn_poc.py`: older standalone synthetic/`.cfile` demonstration with placeholder inference.
- `docs/`: experiment design, parameter validation, deployment readiness, and project status. Some documents are historical experiment logs; verify claims against current code and available artifacts.
- `results/`: generated root-project artifacts; do not assume a documented result exists unless its files are present.
- `external/AWN/`: upstream AWN submodule.
- `external/adversarial-rf/`: adversarial-RF submodule and current paper source. Its local `AGENTS.md`/`CLAUDE.md` applies when editing that submodule.

Initialize dependencies after cloning:

```bash
git submodule update --init --recursive
```

## Canonical paths and boundaries

- Formal pipeline: `experiments/run_full_experiment.py` -> `src/utils/pipeline.py` -> `src/adapters/*`.
- Shared experiment configuration: `src/utils/config.py`.
- Raw GNU Radio capture loader: `src/sensing/iq_source.py` (`complex64` `.cfile`). It is currently used by the standalone script, not wired into the formal pipeline.
- Real AWN implementation used by the adapter: `external/adversarial-rf/models/model.py`.
- Current modular manuscript: `external/adversarial-rf/paper/latex/main.tex` and `paper/latex/sections/*.tex`.
- `external/adversarial-rf/paper/latex/spectral_gated_defense_usenix.tex` is an older monolithic manuscript, not the current paper entry point.
- `external/adversarial-rf/paper/NDSS_PAPER_PLAN_CN.md` is an aspirational plan, not evidence that its systems or experiments exist.

Treat Git submodules as independent repositories. Make root-project changes in the root repository. Modify a submodule only when the task explicitly requires it, follow that submodule's instructions, and report both the submodule commit and the updated root gitlink when relevant.

## Environment and commands

The root NumPy-only demonstration needs the packages in `requirements.txt`:

```bash
python3 -m pip install -r requirements.txt
python3 scripts/sdr_sensing_to_awn_poc.py --demo
```

Formal dry-run smoke test:

```bash
python3 experiments/run_full_experiment.py --dry-run \
  --snr 0 --mod QPSK --attack fgsm --topk 10 \
  --threshold-factor 5 --output-dir /tmp/sdr-awn-smoke
```

Syntax check for root Python code:

```bash
python3 -m compileall -q src experiments scripts
```

Real AWN/attack/Top-K runs additionally require PyTorch and the adversarial-RF dependencies/checkpoint. Never infer that a run used the real backend merely because real-backend flags were requested; verify the backend/status fields in generated output. Use `--help` and `src/utils/config.py` as the current CLI authority because prose examples can lag code.

## Engineering rules

1. Preserve existing public functions and CLI behavior unless a breaking change is explicitly requested.
2. Preserve the documented backward-compatible `naive` segmentation path byte-for-byte. Add improved policies as explicit alternatives.
3. Keep AWN input shape `[N, 2, 128]` unless an experiment explicitly studies a different contract. Keep real/imag ordering and preprocessing policy explicit.
4. Validate IQ arrays at boundaries: one-dimensional complex input, expected dtype/endianness, finite values, clipping/scale, and non-empty captures.
5. Use deterministic seeds for synthetic data, dummy inference, attacks, and sampling. Record the seed and complete configuration with each result.
6. Fail closed in deployment-oriented code when real AWN loading fails. Random dummy logits are acceptable only in an explicitly labeled dry run.
7. Keep detection, segmentation, preprocessing, model inference, attack, defense, and reporting as separate testable stages.
8. Avoid committing datasets, checkpoints, raw captures, generated plots, or bulk experiment outputs unless the task explicitly calls for a small reviewed artifact.
9. Add small deterministic tests under `tests/` for new behavior. Cover malformed inputs, edge cases, backend failure, and backward compatibility.
10. After code changes, run the narrowest relevant checks plus the root syntax check and a smoke run when dependencies permit. State exactly what ran and what could not run.

## Scientific and security integrity

- Deterministic preprocessing is public under a white-box threat model; do not claim security from obscurity.
- Call attacks against only the undefended classifier `oblivious` or `defense-unaware`, not white-box attacks against the complete defended pipeline.
- NDSS robustness claims require defense-aware adaptive evaluation against the complete detector/router/preprocessor/classifier path, using suitable methods such as differentiable surrogates or BPDA, EOT where randomness exists, gradient-free attacks, and multiple restarts.
- Keep one canonical experiment protocol for dataset splits, checkpoint, normalization, attack parameters, seeds, smoothing parameters, and sample selection. Manuscript values and executable configuration must agree.
- Report physically meaningful RF constraints and metrics where applicable, including perturbation norm, perturbation-to-signal ratio, EVM, BER, channel effects, and over-the-air feasibility.
- Distinguish measured facts from proposed architecture. Do not convert planned components or historical summaries into present-tense implementation claims.
- Prefer a passive advisory deployment: retain detected events and escalate uncertain, anomalous, or defense-triggering cases instead of making irreversible autonomous decisions.

## Completion checklist

Before finishing a task:

1. Inspect the actual files and relevant local instructions.
2. Keep changes within the requested repository/submodule scope.
3. Review `git diff` for accidental artifacts, secrets, and unsupported claims.
4. Run relevant deterministic validation.
5. Report changed files, validation commands/results, and remaining limitations.
