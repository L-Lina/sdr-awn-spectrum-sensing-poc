# Phase A validation

**FORMAL DATA VALIDATION: PASS**  (checks: 113, mandatory failed: 0, informational failed: 0)

| ID | Check | Result | Detail |
|---|---|---|---|
| A1.NPU.base_rows | NPU base rows == 2200 | PASS | observed 2200 |
| A1.NPU.attack_rows | NPU attack rows == 52800 | PASS | observed 52800 |
| A1.NPU.defense_rows | NPU defense rows == 211200 | PASS | observed 211200 |
| A1.NPU.manifest_counts | NPU manifest completed_* rows match expected | PASS | manifest completed=(2200,52800,211200) run_status=complete |
| A1.NPU.summary_counts | NPU summary.json counts match CSV row counts | PASS | summary counts={'base_rows': 2200, 'attack_rows': 52800, 'defense_rows': 211200} |
| A2.NPU.config_grid | NPU manifest config grid matches expected | PASS | n_mod=11 n_snr=20 n_idx=10 topk=[10, 20, 30, 40] profiles=24 |
| A2.NPU.profile_count | NPU manifest lists 24 unique attack profiles | PASS | 24 listed |
| A2.NPU.base_keys | NPU base key set == 11x20x10 grid | PASS | missing=0 unexpected=0 |
| A2.NPU.attack_keys | NPU attack key set == base grid x 24 profiles | PASS | missing=0 unexpected=0 |
| A2.NPU.defense_keys | NPU defense key set == attack grid x TopK{10,20,30,40} | PASS | missing=0 unexpected=0 |
| A2.NPU.samples_per_cell | NPU 10 samples per modulation x SNR cell | PASS | cells=220 min=10 max=10 |
| A3.NPU.base_dup | NPU base duplicate keys / duplicate rows = 0 | PASS | dup_keys=0 dup_rows=0 |
| A3.NPU.base_nan | NPU base NaN only in structurally-empty columns | PASS | NaN columns={'missed_signal_samples': 2200, 'false_occupied_samples': 2200}; unexpected={} |
| A3.NPU.base_inf | NPU base Inf = 0 | PASS | Inf columns={} |
| A3.NPU.base_bool | NPU base boolean columns strictly True/False | PASS | dtypes={'clean_correct': 'bool'} |
| A3.NPU.base_status | NPU base status columns all 'ok' | PASS | non-ok counts={'run_status': 0} |
| A3.NPU.base.hilo_backend | NPU base.hilo_backend: single expected backend, no dummy/fallback | PASS | values=['Hailo-8:awn_2016_10a_c5.hef'] |
| A3.NPU.base_strings | NPU base no dummy/fallback/error text in string columns | PASS | hits={} |
| A3.NPU.attack_dup | NPU attack duplicate keys / duplicate rows = 0 | PASS | dup_keys=0 dup_rows=0 |
| A3.NPU.attack_nan | NPU attack NaN = 0 | PASS | NaN columns={} |
| A3.NPU.attack_inf | NPU attack Inf = 0 | PASS | Inf columns={} |
| A3.NPU.attack_bool | NPU attack boolean columns strictly True/False | PASS | dtypes={'clean_correct': 'bool', 'attacked_correct': 'bool', 'attack_success': 'bool'} |
| A3.NPU.attack_status | NPU attack status columns all 'ok' | PASS | non-ok counts={'attack_status': 0, 'run_status': 0} |
| A3.NPU.attack.attack_backend | NPU attack.attack_backend: single expected backend, no dummy/fallback | PASS | values=['external/adversarial-rf/util/adv_attack.py:Model01Wrapper + torchattacks'] |
| A3.NPU.attack.hilo_backend | NPU attack.hilo_backend: single expected backend, no dummy/fallback | PASS | values=['Hailo-8:awn_2016_10a_c5.hef'] |
| A3.NPU.attack_strings | NPU attack no dummy/fallback/error text in string columns | PASS | hits={} |
| A3.NPU.defense_dup | NPU defense duplicate keys / duplicate rows = 0 | PASS | dup_keys=0 dup_rows=0 |
| A3.NPU.defense_nan | NPU defense NaN = 0 | PASS | NaN columns={} |
| A3.NPU.defense_inf | NPU defense Inf = 0 | PASS | Inf columns={} |
| A3.NPU.defense_bool | NPU defense boolean columns strictly True/False | PASS | dtypes={'clean_correct': 'bool', 'attacked_correct': 'bool', 'attack_success': 'bool', 'clean_topk_correct': 'bool', 'clean_degraded': 'bool', 'defended_correct |
| A3.NPU.defense_status | NPU defense status columns all 'ok' | PASS | non-ok counts={'topk_status': 0, 'run_status': 0} |
| A3.NPU.defense.topk_backend | NPU defense.topk_backend: single expected backend, no dummy/fallback | PASS | values=['external/adversarial-rf/util/defense.py:fft_topk_denoise'] |
| A3.NPU.defense.hilo_backend | NPU defense.hilo_backend: single expected backend, no dummy/fallback | PASS | values=['Hailo-8:awn_2016_10a_c5.hef'] |
| A3.NPU.defense_strings | NPU defense no dummy/fallback/error text in string columns | PASS | hits={} |
| A3.NPU.latency_positive | NPU all latency columns finite and > 0 | PASS |  |
| A3.NPU.shape | NPU selected segment length == 128 and true region length == 128 for all rows | PASS | seg_len unique=[np.int64(128)] true_len unique=[np.int64(128)] |
| A3.NPU.label_map | NPU modulation<->label one-to-one | PASS | {'8PSK': 2, 'AM-DSB': 6, 'AM-SSB': 10, 'BPSK': 4, 'CPFSK': 5, 'GFSK': 7, 'PAM4': 8, 'QAM16': 0, 'QAM64': 1, 'QPSK': 9, 'WBFM': 3} |
| A3.NPU.pred_range | NPU predictions in class range [0,10], no missing | PASS |  |
| A3.NPU.hash_format | NPU sha256 fields well-formed (64 hex) | PASS |  |
| A3.NPU.label_consistent | NPU labels consistent across base/attack/defense | PASS |  |
| A3.NPU.correct_flags | NPU *_correct flags agree with pred==label | PASS |  |
| A3.NPU.metric_defs | NPU attack_success = clean_correct & ~attacked_correct; clean_degraded = clean_correct & ~clean_topk_correct; recovered = attack_success & defended_correct | PASS | attack_success=True clean_degraded=True recovered=True |
| A3.NPU.attack_vs_base | NPU attack.clean_pred/clean_correct == base | PASS |  |
| A3.NPU.defense_vs_attack | NPU defense clean/attacked/success flags and adversarial hash == attack table | PASS |  |
| A3.NPU.perturbation | NPU iq_linf/iq_l2/iq_l1 finite and non-negative | PASS |  |
| A3.NPU.latency_recompute | NPU latency_summary.csv recomputed from raw CSV (mean/median/p95/p99/min/max/std, population std ddof=0) | PASS | 12 stages match |
| A3.NPU.summary_latency | NPU summary.json latency == latency_summary.csv | PASS |  |
| A1.CPU.base_rows | CPU base rows == 2200 | PASS | observed 2200 |
| A1.CPU.attack_rows | CPU attack rows == 52800 | PASS | observed 52800 |
| A1.CPU.defense_rows | CPU defense rows == 211200 | PASS | observed 211200 |
| A1.CPU.manifest_counts | CPU manifest completed_* rows match expected | PASS | manifest completed=(2200,52800,211200) run_status=complete |
| A1.CPU.summary_counts | CPU summary.json counts match CSV row counts | PASS | summary counts={'base_rows': 2200, 'attack_rows': 52800, 'defense_rows': 211200} |
| A2.CPU.config_grid | CPU manifest config grid matches expected | PASS | n_mod=11 n_snr=20 n_idx=10 topk=[10, 20, 30, 40] profiles=24 |
| A2.CPU.profile_count | CPU manifest lists 24 unique attack profiles | PASS | 24 listed |
| A2.CPU.base_keys | CPU base key set == 11x20x10 grid | PASS | missing=0 unexpected=0 |
| A2.CPU.attack_keys | CPU attack key set == base grid x 24 profiles | PASS | missing=0 unexpected=0 |
| A2.CPU.defense_keys | CPU defense key set == attack grid x TopK{10,20,30,40} | PASS | missing=0 unexpected=0 |
| A2.CPU.samples_per_cell | CPU 10 samples per modulation x SNR cell | PASS | cells=220 min=10 max=10 |
| A3.CPU.base_dup | CPU base duplicate keys / duplicate rows = 0 | PASS | dup_keys=0 dup_rows=0 |
| A3.CPU.base_nan | CPU base NaN only in structurally-empty columns | PASS | NaN columns={'missed_signal_samples': 2200, 'false_occupied_samples': 2200}; unexpected={} |
| A3.CPU.base_inf | CPU base Inf = 0 | PASS | Inf columns={} |
| A3.CPU.base_bool | CPU base boolean columns strictly True/False | PASS | dtypes={'clean_correct': 'bool'} |
| A3.CPU.base_status | CPU base status columns all 'ok' | PASS | non-ok counts={'run_status': 0} |
| A3.CPU.base.inference_backend | CPU base.inference_backend: single expected backend, no dummy/fallback | PASS | values=['external/adversarial-rf/models/model.py:AWN'] |
| A3.CPU.base_strings | CPU base no dummy/fallback/error text in string columns | PASS | hits={} |
| A3.CPU.attack_dup | CPU attack duplicate keys / duplicate rows = 0 | PASS | dup_keys=0 dup_rows=0 |
| A3.CPU.attack_nan | CPU attack NaN = 0 | PASS | NaN columns={} |
| A3.CPU.attack_inf | CPU attack Inf = 0 | PASS | Inf columns={} |
| A3.CPU.attack_bool | CPU attack boolean columns strictly True/False | PASS | dtypes={'clean_correct': 'bool', 'attacked_correct': 'bool', 'attack_success': 'bool'} |
| A3.CPU.attack_status | CPU attack status columns all 'ok' | PASS | non-ok counts={'attack_status': 0, 'run_status': 0} |
| A3.CPU.attack.attack_backend | CPU attack.attack_backend: single expected backend, no dummy/fallback | PASS | values=['external/adversarial-rf/util/adv_attack.py:Model01Wrapper + torchattacks'] |
| A3.CPU.attack.inference_backend | CPU attack.inference_backend: single expected backend, no dummy/fallback | PASS | values=['external/adversarial-rf/models/model.py:AWN'] |
| A3.CPU.attack_strings | CPU attack no dummy/fallback/error text in string columns | PASS | hits={} |
| A3.CPU.defense_dup | CPU defense duplicate keys / duplicate rows = 0 | PASS | dup_keys=0 dup_rows=0 |
| A3.CPU.defense_nan | CPU defense NaN = 0 | PASS | NaN columns={} |
| A3.CPU.defense_inf | CPU defense Inf = 0 | PASS | Inf columns={} |
| A3.CPU.defense_bool | CPU defense boolean columns strictly True/False | PASS | dtypes={'clean_correct': 'bool', 'attacked_correct': 'bool', 'attack_success': 'bool', 'clean_topk_correct': 'bool', 'clean_degraded': 'bool', 'defended_correct |
| A3.CPU.defense_status | CPU defense status columns all 'ok' | PASS | non-ok counts={'topk_status': 0, 'run_status': 0} |
| A3.CPU.defense.topk_backend | CPU defense.topk_backend: single expected backend, no dummy/fallback | PASS | values=['external/adversarial-rf/util/defense.py:fft_topk_denoise'] |
| A3.CPU.defense.inference_backend | CPU defense.inference_backend: single expected backend, no dummy/fallback | PASS | values=['external/adversarial-rf/models/model.py:AWN'] |
| A3.CPU.defense_strings | CPU defense no dummy/fallback/error text in string columns | PASS | hits={} |
| A3.CPU.latency_positive | CPU all latency columns finite and > 0 | PASS |  |
| A3.CPU.shape | CPU selected segment length == 128 and true region length == 128 for all rows | PASS | seg_len unique=[np.int64(128)] true_len unique=[np.int64(128)] |
| A3.CPU.label_map | CPU modulation<->label one-to-one | PASS | {'8PSK': 2, 'AM-DSB': 6, 'AM-SSB': 10, 'BPSK': 4, 'CPFSK': 5, 'GFSK': 7, 'PAM4': 8, 'QAM16': 0, 'QAM64': 1, 'QPSK': 9, 'WBFM': 3} |
| A3.CPU.pred_range | CPU predictions in class range [0,10], no missing | PASS |  |
| A3.CPU.hash_format | CPU sha256 fields well-formed (64 hex) | PASS |  |
| A3.CPU.label_consistent | CPU labels consistent across base/attack/defense | PASS |  |
| A3.CPU.correct_flags | CPU *_correct flags agree with pred==label | PASS |  |
| A3.CPU.metric_defs | CPU attack_success = clean_correct & ~attacked_correct; clean_degraded = clean_correct & ~clean_topk_correct; recovered = attack_success & defended_correct | PASS | attack_success=True clean_degraded=True recovered=True |
| A3.CPU.attack_vs_base | CPU attack.clean_pred/clean_correct == base | PASS |  |
| A3.CPU.defense_vs_attack | CPU defense clean/attacked/success flags and adversarial hash == attack table | PASS |  |
| A3.CPU.perturbation | CPU iq_linf/iq_l2/iq_l1 finite and non-negative | PASS |  |
| A3.CPU.latency_recompute | CPU latency_summary.csv recomputed from raw CSV (mean/median/p95/p99/min/max/std, population std ddof=0) | PASS | 12 stages match |
| A3.CPU.summary_latency | CPU summary.json latency == latency_summary.csv | PASS |  |
| A4.rows | CPU/NPU base merge on (modulation,snr,sample_index) is one-to-one, 2200 rows | PASS | merged=2200 |
| A4.fields | CPU/NPU sensing fields identical row-by-row (label, seed, true_start/end, selected segment start/end, detected region, clean_input_sha256, ...) | PASS | mismatch counts={'label': 0, 'seed': 0, 'true_start': 0, 'true_end': 0, 'region_count': 0, 'selected_segment_start': 0, 'selected_segment_end': 0, 'detected_reg |
| A6.CPU.dataset_hash | CPU manifest dataset_sha256 == recomputed sha256 of data/RML2016.10a_dict.pkl | PASS | manifest=b29ccc25b00d0718cd3b70ffa9158662ec83f6d9b63ffd845c7bcbe3b3096e8c |
| A6.CPU.checkpoint_hash | CPU manifest checkpoint_sha256 == recomputed sha256 of external/adversarial-rf/2016.10a_AWN.pkl | PASS | manifest=8af0458f2570c465b5bb0ebad00817944f8171d888cd7cff1324ecb258820695 |
| A6.NPU.dataset_hash | NPU manifest dataset_sha256 == recomputed sha256 of data/RML2016.10a_dict.pkl | PASS | manifest=b29ccc25b00d0718cd3b70ffa9158662ec83f6d9b63ffd845c7bcbe3b3096e8c |
| A6.NPU.checkpoint_hash | NPU manifest checkpoint_sha256 == recomputed sha256 of external/adversarial-rf/2016.10a_AWN.pkl | PASS | manifest=8af0458f2570c465b5bb0ebad00817944f8171d888cd7cff1324ecb258820695 |
| A6.NPU.hef_hash | NPU manifest hef_sha256 == recomputed sha256 of HEF | PASS | manifest=6e54266f7e5931e432d4e5e67cd8e4a6e14863c16a94ff02a1946d21abc73dbb |
| A6.NPU.hef_backend_col | NPU per-row hilo_backend names the manifest HEF file | PASS |  |
| A6.CPU.no_hef_hash | CPU manifest has no hef_sha256 (CPU backend does not deploy a HEF) | PASS | CPU manifest contains 'hef_path' in config only as inherited config text |
| A6.CPU.backend | CPU manifest inference_backend == 'PyTorch CPU AWN' | PASS | PyTorch CPU AWN |
| A6.NPU.threat_model | NPU manifest threat_model states PyTorch-surrogate generation and quantized Hailo-8 evaluation | PASS | A0 digital classifier-input attack; adversarial examples generated with differentiable PyTorch AWN surrogate and evaluated on deployed quantized Hailo-8 AWN. |
| A6.CPU.threat_model | CPU manifest threat_model states generation and evaluation on original PyTorch AWN | PASS | A0 digital classifier-input attack; adversarial examples generated and evaluated on the original differentiable PyTorch AWN. |
| A6.config_equal | CPU and NPU manifest 'config' blocks identical | PASS |  |
| A6.hashes_equal | CPU and NPU manifests share dataset/checkpoint hashes | PASS |  |
| A6.config_on_disk | configs/hailo_full_matrix.json (current working tree) == manifest config | PASS | config file is untracked in git; compared for reference |
| A6.seed | manifest sensing.base_seed == 42 and per-row seeds identical across CPU/NPU | PASS | base_seed CPU=42 NPU=42; unique per-row seeds=2200 |
| A3.NPU.nohup.log | NPU nohup.log: no traceback/exception/fallback/dummy/error/failed lines | PASS | lines=563129 flagged=0 samples=[] |
| A3.CPU.nohup.log | CPU nohup.log: no traceback/exception/fallback/dummy/error/failed lines | PASS | lines=563139 flagged=0 samples=[] |
| A3.CPU.run.log | CPU run.log: no traceback/exception/fallback/dummy/error/failed lines | PASS | lines=563138 flagged=0 samples=[] |

## Notes
- NPU base_results: columns ['false_occupied_samples', 'missed_signal_samples'] are empty in all 2200 rows (not used in any analysis).
- CPU base_results: columns ['false_occupied_samples', 'missed_signal_samples'] are empty in all 2200 rows (not used in any analysis).
- Both manifests carry config.experiment_name='hailo_full_matrix_20260919' (name inherited from the shared config file); backends are distinguished by the results directory names and by the manifest fields 'inference_backend' / 'hef_sha256'.
- Manifests record base_seed=42 and per-row seeds but no git commit hash / software versions.

## Clean CPU vs NPU
```
{
  "n": 2200,
  "prediction_agreement_count": 1996,
  "prediction_agreement_pct": 90.72727272727273,
  "disagreement_count": 204,
  "both_correct": 1172,
  "cpu_correct_npu_wrong": 126,
  "cpu_wrong_npu_correct": 30,
  "both_wrong": 872,
  "cpu_clean_accuracy_pct": 59.0,
  "npu_clean_accuracy_pct": 54.63636363636364,
  "accuracy_diff_pp_npu_minus_cpu": -4.363636363636358
}
```

## CPU/NPU adversarial-example identity (informational)
```
{
  "attack_rows_merged": 52800,
  "adversarial_sha256_identical_rows": 35384,
  "adversarial_sha256_differing_rows": 17416,
  "iq_linf_max_abs_diff": 0.0010769548825918635,
  "iq_l2_max_abs_diff": 0.0172312612610605,
  "clean_pred_identical_in_attack_table": 47904,
  "differing_by_attack_profile": "{('autoattack', 'eps0.03_standard'): 2022, ('pgd', 'eps0.005'): 2200, ('pgd', 'eps0.01'): 2200, ('pgd', 'eps0.03'): 2200, ('rfgsm', 'eps0.03_default'): 2200, ('tpgd', 'eps0.03_default'): 2200, ('vmifgsm', 'eps0.03_default'): 2200, ('vnifgsm', 'eps0.03_default'): 2194}"
}
```