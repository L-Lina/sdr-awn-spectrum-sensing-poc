# sdr-awn-spectrum-sensing-poc

## Purpose

Proof-of-concept for connecting the AWN AMC (Automatic Modulation Classification)
model to an SDR / GNU Radio spectrum sensing pipeline. This stage stays hardware-free
and torch-free: it uses numpy to simulate a GNU Radio IQ capture, runs a simple
energy-detection + windowing front end, and produces tensors shaped for AWN's
`[N, 2, 128]` input (batch, real/imag, time samples). Real AWN inference is stubbed
out behind a placeholder function so the rest of the pipeline can be developed and
tested independently of the model.

## Pipeline

```
complex IQ stream (synthetic or captured_iq.cfile)
  -> energy detection            (energy_detect)
  -> occupied region extraction  (extract_occupied_regions)
  -> fixed-length windowing      (segment_regions)
  -> per-segment normalization   (normalize_segments)
  -> AWN input [N, 2, window]    (to_awn_input)
  -> AWN inference placeholder   (run_awn_inference)
```

## How to run the demo

```bash
pip install -r requirements.txt

python scripts/sdr_sensing_to_awn_poc.py --demo
```

Useful flags:

- `--demo` - generate a synthetic noise + burst IQ stream instead of reading a file
- `--input captured_iq.cfile` - read real IQ samples instead of `--demo`
- `--window-size 128` - segment length in samples (also the energy-detection window); AWN expects 128
- `--threshold-factor 5.0` - energy detection threshold as a multiple of the median noise power
- `--output data/awn_input.npy` - save the resulting `[N, 2, window]` tensor to disk

Example with output saved:

```bash
python scripts/sdr_sensing_to_awn_poc.py --demo --output data/awn_input.npy
```

## Connecting a real GNU Radio capture later

1. Build a flowgraph: `UHD: USRP Source -> File Sink` with the File Sink's output type
   set to **complex64** (`gr_complex`), writing to e.g. `captured_iq.cfile`.
2. Run this script against that capture instead of `--demo`:

   ```bash
   python scripts/sdr_sensing_to_awn_poc.py --input captured_iq.cfile
   ```

   Internally this is just `np.fromfile("captured_iq.cfile", dtype=np.complex64)` -
   no GNU Radio dependency at runtime, only a matching byte layout.
3. A future streaming variant (not implemented yet) would replace the file read with
   a GNU Radio `ZMQ PUB Sink -> zmq.SUB` socket, accumulating samples into a rolling
   buffer and re-running the same `energy_detect` / `segment_regions` / `to_awn_input`
   functions on each new chunk.

## How this connects to AWN

`run_awn_inference(x)` in `scripts/sdr_sensing_to_awn_poc.py` is currently a placeholder:
it validates the `[N, 2, window]` shape and returns random logits, with no torch
dependency. The intended follow-up is to swap its body for a real AWN model load
(`torch.load` + `model.eval()` + forward pass) once this front-end pipeline is
validated, and to add adversarial attack / Top-K defense hooks around it to match
the defender architecture used elsewhere in this project.
