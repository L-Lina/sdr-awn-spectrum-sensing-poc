"""
Adversarial training of the AWN classifier for the defense recovery comparison.

Why this exists next to external/adversarial-rf/adv_train.py
------------------------------------------------------------
The existing checkpoint (external/adversarial-rf/awn_fpga/2016.10a_AWN_at.pkl, produced by adv_train.py)
splits all SNR >= 0 RadioML rows randomly, so formal evaluation samples (sample indices of
configs/hailo_full_matrix.json) can be training samples, and its attacks use eps=0.1 / EAD settings that are
not the formal profiles. This script trains the same way (warm start, mixed clean/adversarial loss) but:
  * excludes every formal evaluation sample (all modulations x SNRs x formal sample_indices) from train/val,
  * takes the training attacks from formal profiles named in the training config (parameters are read from
    the formal config, never re-typed),
  * is resumable and writes a manifest with all hashes and the exact split.

Reused, not re-implemented: AWN architecture and configuration (src/adapters/awn_adapter.py), RadioML loader
(src/sensing/radioml_source.py), attack construction and the [0,1] min-max mapping used by the formal attack
path (src/adapters/attack_adapter.py: _build_torchattacks, TemperatureLogitsWrapper, Model01Wrapper,
iq_to_ta_input_minmax / ta_output_to_iq_minmax).

Training step (per batch): the attack (round-robin over train_attacks; torch seeded per global step) is
generated against the current model in eval mode with true labels; rows of clean_only_modulations keep their
clean input in the adversarial stream (as adv_train.py does); then in train mode
    loss = (1 - alpha) * CE(model(x_clean), y) + alpha * CE(model(x_adv), y) + regu(clean) + regu(adv).
Attacks run batched during training; this is a training-data generator, not an evaluation of attack
semantics. Validation each epoch: clean accuracy and robust accuracy (robust_val_attack) on the validation
split; selection metric = val_weight_clean * clean + (1 - val_weight_clean) * robust. The best epoch is saved as
a plain AWN state_dict (loadable by AWNModelAdapter) plus <stem>.config.json describing the split.

Outputs (out dir): best_model.pkl, best_model.config.json, state_last.pt (resume), training_log.csv,
split_indices.npz, manifest.json.

Run (no training is started by --dry-run):
    .venv/bin/python experiments/train_adversarial_awn.py --dry-run
    nohup .venv/bin/python experiments/train_adversarial_awn.py --device cuda > at_train.log 2>&1 &   (GPU host)
    quick functional check: --max-batches-per-epoch 2 (marked functional_check_only; never leakage_free)
    resume: .venv/bin/python experiments/train_adversarial_awn.py --resume results/adversarial_training_<ts>
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_CONFIG = "configs/adversarial_training_awn.json"
LOG_FIELDS = ["epoch", "lr", "train_loss", "train_loss_clean", "train_loss_adv", "train_clean_acc",
              "val_clean_acc", "val_robust_acc", "val_weighted", "best", "seconds"]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def resolve_profile(formal_cfg: dict, attack: str, profile_id: str) -> dict:
    for entry in formal_cfg["attacks"]:
        if str(entry["name"]).lower() != attack:
            continue
        for prof in entry["profiles"]:
            if prof["id"] == profile_id:
                return {"attack": attack, "profile_id": profile_id, "eps": float(prof.get("eps", 0.03)),
                        "attack_params": dict(prof.get("params", {})), "cw_c": float(prof.get("cw_c", 1.0)),
                        "cw_steps": int(prof.get("cw_steps", 20)), "cw_lr": float(prof.get("cw_lr", 0.01))}
    raise SystemExit(f"[ABORT] formal config has no profile {attack}/{profile_id}")


def build_split(tcfg: dict, formal_cfg: dict, data: dict):
    from src.sensing.radioml_source import RML2016_10A_CLASSES
    mods = list(formal_cfg["modulations"])
    snrs = sorted(s for s in {s for _, s in data} if tcfg["snr_min"] is None or s >= tcfg["snr_min"])
    excluded = set(formal_cfg["sample_indices"]) if tcfg["exclude_formal_eval_samples"] else set()
    keys = []
    for m in mods:
        for s in snrs:
            n = data[(m, s)].shape[0]
            keys += [(m, s, i) for i in range(n) if i not in excluded]
    perm = np.random.default_rng(tcfg["seed"]).permutation(len(keys))
    n_val = int(len(keys) * tcfg["val_ratio"])
    train_idx, val_idx = perm[: len(keys) - n_val], perm[len(keys) - n_val:]
    x = np.stack([data[(m, s)][i] for m, s, i in keys]).astype(np.float32)
    y = np.asarray([RML2016_10A_CLASSES[m] for m, _, _ in keys], dtype=np.int64)
    return keys, x, y, train_idx, val_idx, snrs, sorted(excluded)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--resume", default=None, help="existing output dir; continues after the last saved epoch")
    ap.add_argument("--dry-run", action="store_true", help="resolve config, split and profiles; do not train")
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto",
                    help="auto = cuda if available, else cpu (checkpoints are always saved as CPU tensors)")
    ap.add_argument("--max-batches-per-epoch", type=int, default=None,
                    help="limit for a quick functional check (recorded in the manifest; not a formal model)")
    args = ap.parse_args()

    tcfg_path = REPO_ROOT / args.config
    tcfg = json.loads(tcfg_path.read_text())
    fcfg_path = REPO_ROOT / tcfg["formal_config"]
    formal_cfg = json.loads(fcfg_path.read_text())
    dataset_path = REPO_ROOT / (tcfg["dataset_path"] or formal_cfg["dataset_path"])
    warm_path = REPO_ROOT / (tcfg["warm_start_checkpoint"] or formal_cfg["checkpoint_path"])
    train_profiles = [resolve_profile(formal_cfg, a["attack"], a["profile_id"]) for a in tcfg["train_attacks"]]
    val_profile = resolve_profile(formal_cfg, tcfg["robust_val_attack"]["attack"],
                                  tcfg["robust_val_attack"]["profile_id"])

    from src.sensing.radioml_source import RML2016_10A_CLASSES, load_radioml_dict
    data = load_radioml_dict(str(dataset_path))
    keys, X, Y, train_idx, val_idx, snrs, excluded = build_split(tcfg, formal_cfg, data)
    clean_only = {RML2016_10A_CLASSES[m] for m in tcfg["clean_only_modulations"]}
    plan = {"n_rows": len(keys), "n_train": int(len(train_idx)), "n_val": int(len(val_idx)), "snrs": snrs,
            "excluded_sample_indices": excluded, "train_profiles": train_profiles, "val_profile": val_profile,
            "batches_per_epoch": int(np.ceil(len(train_idx) / tcfg["batch_size"]))}
    log(f"[plan] {json.dumps(plan)}")
    if args.dry_run:
        print(json.dumps({"config": tcfg, "plan": plan}, indent=2))
        return

    import torch
    import torch.nn as nn
    import src.adapters.attack_adapter as am
    from src.adapters.awn_adapter import _AWN_2016_10A_CFG, _real_AWN_cls
    if _real_AWN_cls is None or am._torchattacks is None or am._Model01Wrapper is None:
        raise SystemExit("[ABORT] torch / AWN / torchattacks unavailable")
    if tcfg.get("torch_num_threads"):
        torch.set_num_threads(int(tcfg["torch_num_threads"]))
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("[ABORT] --device cuda requested but CUDA is not available")
    device = torch.device("cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())
                          else "cpu")
    log(f"[device] {device}")

    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out = (REPO_ROOT / (args.resume or args.out_dir or f"results/adversarial_training_{ts}")).resolve()
    hashes = {"training_config_sha256": sha256_file(tcfg_path), "formal_config_sha256": sha256_file(fcfg_path),
              "dataset_sha256": sha256_file(dataset_path), "warm_start_sha256": sha256_file(warm_path),
              "this_script_sha256": sha256_file(Path(__file__).resolve())}
    split_digest = hashlib.sha256(np.ascontiguousarray(train_idx).tobytes()
                                  + np.ascontiguousarray(val_idx).tobytes()).hexdigest()
    if args.resume:
        man = json.loads((out / "manifest.json").read_text())
        for k in ("training_config_sha256", "formal_config_sha256", "dataset_sha256", "warm_start_sha256"):
            if man["hashes"][k] != hashes[k]:
                raise SystemExit(f"[ABORT] --resume: {k} differs from the original run")
        if man["split_digest"] != split_digest:
            raise SystemExit("[ABORT] --resume: train/val split differs from the original run")
    else:
        out.mkdir(parents=True, exist_ok=False)
        np.savez(out / "split_indices.npz", train_idx=train_idx, val_idx=val_idx)
        git = subprocess.run(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], capture_output=True, text=True)
        man = {"script": "experiments/train_adversarial_awn.py", "config": tcfg, "plan": plan, "hashes": hashes,
               "split_digest": split_digest, "git_head": git.stdout.strip(), "torch_version": torch.__version__,
               "device": str(device),
               "started_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
               "max_batches_per_epoch": args.max_batches_per_epoch,
               "functional_check_only": args.max_batches_per_epoch is not None, "status": "running"}
    (out / "manifest.json").write_text(json.dumps(man, indent=2, default=str))

    model = _real_AWN_cls(**_AWN_2016_10A_CFG)
    model.load_state_dict(torch.load(warm_path, map_location="cpu"))
    model.to(device)
    wm = am._Model01Wrapper(model)
    opt = torch.optim.Adam(model.parameters(), lr=tcfg["lr"])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=tcfg["lr_plateau_factor"],
                                                       patience=tcfg["lr_plateau_patience"])
    ce = nn.CrossEntropyLoss()
    start_epoch, best, no_improve = 1, -1.0, 0
    state_path = out / "state_last.pt"
    if args.resume and state_path.exists():
        st = torch.load(state_path, map_location="cpu", weights_only=False)  # own file: optimizer + scheduler state
        man.setdefault("resumes", []).append({"utc": _dt.datetime.now(_dt.timezone.utc).isoformat(), "device": str(device)})
        man["status"] = "running"
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["optimizer"])
        sched.load_state_dict(st["scheduler"])
        start_epoch, best, no_improve = st["epoch"] + 1, st["best"], st["no_improve"]
        log(f"[resume] continuing at epoch {start_epoch} (best={best:.4f})")

    def cpu_state() -> dict:
        return {k: v.detach().cpu() for k, v in model.state_dict().items()}

    def regu_of(r):
        return r if torch.is_tensor(r) else sum(r)

    def attack(xb: torch.Tensor, yb: torch.Tensor, prof: dict, seed: int) -> torch.Tensor:
        model.eval()
        torch.manual_seed(seed)
        x_ta, a, b = am._iq_to_ta_input_minmax(xb)
        wm.set_minmax(a, b)
        try:
            atk = am._build_torchattacks(prof["attack"], am.TemperatureLogitsWrapper(wm, 1.0), prof["eps"],
                                         cw_c=prof["cw_c"], cw_steps=prof["cw_steps"], cw_lr=prof["cw_lr"],
                                         attack_params=dict(prof["attack_params"]))
            x_adv = am._ta_output_to_iq_minmax(atk(x_ta, yb), a, b).detach()
        finally:
            wm.clear_minmax()
        return x_adv

    def evaluate(idx: np.ndarray, epoch: int) -> tuple:
        model.eval()
        n_clean = n_rob = 0
        bs = tcfg["batch_size"]
        for s in range(0, len(idx), bs):
            bi = idx[s:s + bs]
            xb, yb = torch.from_numpy(X[bi]).to(device), torch.from_numpy(Y[bi]).to(device)
            with torch.no_grad():
                n_clean += int((model(xb)[0].argmax(1) == yb).sum())
            xa = attack(xb, yb, val_profile, tcfg["seed"] * 1_000_003 + 7 * epoch + s)
            with torch.no_grad():
                n_rob += int((model(xa)[0].argmax(1) == yb).sum())
        return n_clean / len(idx), n_rob / len(idx)

    log_path = out / "training_log.csv"
    if not log_path.exists():
        with log_path.open("w", newline="") as f:
            csv.DictWriter(f, fieldnames=LOG_FIELDS).writeheader()
    bs = tcfg["batch_size"]
    alpha = float(tcfg["alpha_adversarial"])
    clean_only_t = torch.tensor(sorted(clean_only), dtype=torch.long, device=device)
    for epoch in range(start_epoch, tcfg["epochs"] + 1):
        t0 = time.time()
        order = train_idx[np.random.default_rng(tcfg["seed"] + epoch).permutation(len(train_idx))]
        n_batches = int(np.ceil(len(order) / bs))
        if args.max_batches_per_epoch:
            n_batches = min(n_batches, args.max_batches_per_epoch)
        sums = {"loss": 0.0, "clean": 0.0, "adv": 0.0, "acc": 0.0}
        for bnum in range(n_batches):
            bi = order[bnum * bs:(bnum + 1) * bs]
            xb, yb = torch.from_numpy(X[bi]).to(device), torch.from_numpy(Y[bi]).to(device)
            step = (epoch - 1) * 1_000_000 + bnum
            prof = train_profiles[bnum % len(train_profiles)]
            xa = attack(xb, yb, prof, tcfg["seed"] * 1_000_003 + step)
            keep = torch.isin(yb, clean_only_t)
            if keep.any():
                xa[keep] = xb[keep]
            model.train()
            lc, rc = model(xb)
            la, ra = model(xa)
            loss_c, loss_a = ce(lc, yb), ce(la, yb)
            loss = (1 - alpha) * loss_c + alpha * loss_a + regu_of(rc) + regu_of(ra)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sums["loss"] += float(loss)
            sums["clean"] += float(loss_c)
            sums["adv"] += float(loss_a)
            sums["acc"] += float((lc.argmax(1) == yb).float().mean())
            if bnum % 50 == 0:
                log(f"[epoch {epoch}] batch {bnum}/{n_batches} loss={float(loss):.4f} attack={prof['attack']}")
        val_clean, val_rob = evaluate(val_idx, epoch)
        weighted = tcfg["val_weight_clean"] * val_clean + (1 - tcfg["val_weight_clean"]) * val_rob
        sched.step(weighted)
        is_best = weighted > best
        if is_best:
            best, no_improve = weighted, 0
            torch.save(cpu_state(), out / "best_model.pkl.tmp")
            os.replace(out / "best_model.pkl.tmp", out / "best_model.pkl")
            (out / "best_model.config.json").write_text(json.dumps({
                "split_method": "train_adversarial_awn", "formal_eval_samples_excluded":
                    bool(tcfg["exclude_formal_eval_samples"]), "excluded_sample_indices": excluded,
                "snr_min": tcfg["snr_min"], "seed": tcfg["seed"], "val_ratio": tcfg["val_ratio"],
                "train_profiles": train_profiles, "alpha_adversarial": alpha, "best_epoch": epoch,
                "best_val_weighted": weighted, "warm_start_sha256": hashes["warm_start_sha256"],
                "split_digest": split_digest, "functional_check_only": args.max_batches_per_epoch is not None},
                indent=2))
        else:
            no_improve += 1
        row = {"epoch": epoch, "lr": opt.param_groups[0]["lr"], "train_loss": sums["loss"] / n_batches,
               "train_loss_clean": sums["clean"] / n_batches, "train_loss_adv": sums["adv"] / n_batches,
               "train_clean_acc": sums["acc"] / n_batches, "val_clean_acc": val_clean, "val_robust_acc": val_rob,
               "val_weighted": weighted, "best": is_best, "seconds": time.time() - t0}
        with log_path.open("a", newline="") as f:
            csv.DictWriter(f, fieldnames=LOG_FIELDS).writerow(row)
        torch.save({"model": cpu_state(), "optimizer": opt.state_dict(), "scheduler": sched.state_dict(),
                    "epoch": epoch, "best": best, "no_improve": no_improve}, out / "state_last.pt.tmp")
        os.replace(out / "state_last.pt.tmp", state_path)
        log(f"[epoch {epoch}] val_clean={val_clean:.4f} val_robust={val_rob:.4f} weighted={weighted:.4f}"
            f"{' (best)' if is_best else ''}")
        if no_improve >= tcfg["early_stop_patience"]:
            log(f"[early stop] no improvement for {no_improve} epochs")
            break
    man.update(status="complete", finished_utc=_dt.datetime.now(_dt.timezone.utc).isoformat(), best_val_weighted=best,
               best_model_sha256=sha256_file(out / "best_model.pkl") if (out / "best_model.pkl").exists() else None)
    (out / "manifest.json").write_text(json.dumps(man, indent=2, default=str))
    log(f"[done] {out}")


if __name__ == "__main__":
    main()
