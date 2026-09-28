"""
Hailo-8 inference adapter for the deployment-equivalent AWN HEF.

Input contract:
    numpy float32 [N, 2, 128]

Output contract:
    numpy float32 [N, 11]

The host-facing Hailo vstreams use FLOAT32, so HailoRT performs the
HEF-defined input quantization and output dequantization automatically.
Do not manually cast AWN inputs to uint8.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np


def _import_hailo_platform():
    try:
        import hailo_platform as hpf
        return hpf
    except ModuleNotFoundError:
        # Raspberry Pi HailoRT Debian package location.
        system_dist = "/usr/lib/python3/dist-packages"
        if system_dist not in sys.path:
            sys.path.append(system_dist)
        import hailo_platform as hpf
        return hpf


class HailoAWNAdapter:
    """
    Persistent HailoRT AWN inference backend.

    The VDevice, configured network, vstreams, and activation context are
    created once and reused across infer() calls.
    """

    def __init__(self, hef_path: str) -> None:
        self.hef_path = str(Path(hef_path).expanduser().resolve())

        if not Path(self.hef_path).is_file():
            raise FileNotFoundError(f"HEF not found: {self.hef_path}")

        self.backend_name = "Hailo-8:awn_2016_10a_c5.hef"
        self.status = "initializing"
        self.notes = ""

        self._hpf = _import_hailo_platform()

        self._vdevice_cm = None
        self._vdevice = None
        self._pipeline_cm = None
        self._pipeline = None
        self._activation_cm = None

        try:
            self.hef = self._hpf.HEF(self.hef_path)

            input_infos = self.hef.get_input_vstream_infos()
            output_infos = self.hef.get_output_vstream_infos()

            if len(input_infos) != 1:
                raise RuntimeError(
                    f"Expected exactly 1 HEF input vstream, got {len(input_infos)}"
                )

            if len(output_infos) != 1:
                raise RuntimeError(
                    f"Expected exactly 1 HEF output vstream, got {len(output_infos)}"
                )

            self.input_info = input_infos[0]
            self.output_info = output_infos[0]

            if tuple(self.input_info.shape) != (2, 128, 1):
                raise RuntimeError(
                    f"Unexpected HEF input shape {self.input_info.shape}; "
                    "expected (2, 128, 1)"
                )

            if tuple(self.output_info.shape) != (1, 1, 11):
                raise RuntimeError(
                    f"Unexpected HEF output shape {self.output_info.shape}; "
                    "expected (1, 1, 11)"
                )

            self._vdevice_cm = self._hpf.VDevice()
            self._vdevice = self._vdevice_cm.__enter__()

            config = self._hpf.ConfigureParams.create_from_hef(
                self.hef,
                interface=self._hpf.HailoStreamInterface.PCIe,
            )

            network_groups = self._vdevice.configure(self.hef, config)

            if len(network_groups) != 1:
                raise RuntimeError(
                    f"Expected exactly 1 configured network group, got "
                    f"{len(network_groups)}"
                )

            self.network_group = network_groups[0]
            self.network_group_params = self.network_group.create_params()

            self.input_params = (
                self._hpf.InputVStreamParams.make_from_network_group(
                    self.network_group,
                    quantized=False,
                    format_type=self._hpf.FormatType.FLOAT32,
                )
            )

            self.output_params = (
                self._hpf.OutputVStreamParams.make_from_network_group(
                    self.network_group,
                    quantized=False,
                    format_type=self._hpf.FormatType.FLOAT32,
                )
            )

            self._pipeline_cm = self._hpf.InferVStreams(
                self.network_group,
                self.input_params,
                self.output_params,
            )
            self._pipeline = self._pipeline_cm.__enter__()

            self._activation_cm = self.network_group.activate(
                self.network_group_params
            )
            self._activation_cm.__enter__()

            self.status = "ok"
            self.notes = (
                f"Loaded HEF '{self.hef_path}' on Hailo-8 with FLOAT32 "
                "host vstreams and HEF-defined quantization/dequantization."
            )

        except Exception:
            self.close()
            raise

    def infer(
        self,
        x: np.ndarray,
        n_classes: int = 11,
        seed=None,
    ) -> Tuple[np.ndarray, Dict[str, str]]:
        del seed

        x = np.asarray(x)

        if x.ndim != 3 or x.shape[1:] != (2, 128):
            raise ValueError(
                f"Hailo AWN expects input [N, 2, 128], got {x.shape}"
            )

        if n_classes != 11:
            raise ValueError(
                f"Hailo AWN HEF has exactly 11 classes, got n_classes={n_classes}"
            )

        if x.dtype != np.float32:
            x = x.astype(np.float32, copy=False)

        if not np.isfinite(x).all():
            raise ValueError("Hailo AWN input contains NaN or Inf")

        # HEF host layout: [N, 2, 128, 1]
        x_hailo = x[..., np.newaxis]

        result = self._pipeline.infer(
            {self.input_info.name: x_hailo}
        )

        logits = np.asarray(
            result[self.output_info.name],
            dtype=np.float32,
        )

        logits = logits.reshape(x.shape[0], -1)

        if logits.shape != (x.shape[0], 11):
            raise RuntimeError(
                f"Unexpected Hailo AWN logits shape {logits.shape}; "
                f"expected ({x.shape[0]}, 11)"
            )

        if not np.isfinite(logits).all():
            raise RuntimeError("Hailo AWN output contains NaN or Inf")

        meta = {
            "awn_backend": self.backend_name,
            "awn_status": "ok",
            "awn_notes": self.notes,
        }

        print(
            f"[hailo_awn_adapter] backend={self.backend_name} "
            f"status=ok input={x.shape} logits={logits.shape}"
        )

        return logits, meta

    def close(self) -> None:
        # Release in reverse construction order.
        if self._activation_cm is not None:
            try:
                self._activation_cm.__exit__(None, None, None)
            except Exception:
                pass
            self._activation_cm = None

        if self._pipeline_cm is not None:
            try:
                self._pipeline_cm.__exit__(None, None, None)
            except Exception:
                pass
            self._pipeline_cm = None
            self._pipeline = None

        if self._vdevice_cm is not None:
            try:
                self._vdevice_cm.__exit__(None, None, None)
            except Exception:
                pass
            self._vdevice_cm = None
            self._vdevice = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False
