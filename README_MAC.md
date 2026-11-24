# StreamDiffusion on Mac (Apple Silicon / MPS)

Fork of [cumulo-autumn/StreamDiffusion](https://github.com/cumulo-autumn/StreamDiffusion) that runs realtime webcam img2img on Apple Silicon using PyTorch MPS (no CUDA / TensorRT / xformers).

## Changes from upstream
- `src/streamdiffusion/pipeline.py`: replaced `torch.cuda.Event` timing with device-agnostic timing (`torch.mps.synchronize()` on MPS).
- `mac_realtime.py`: webcam realtime img2img viewer + benchmark (sd-turbo + TAESD, fp16, MPS).
- `requirements-mac.txt`: pinned deps that work with diffusers 0.24.0.

## Setup
```bash
uv venv -p 3.10 .venv && source .venv/bin/activate
uv pip install -r requirements-mac.txt
uv pip install -e . --no-deps
```

## Run
```bash
python mac_realtime.py --size 384 --t 32 --prompt "anime illustration, cel shading"
python mac_realtime.py --size 256 --t 32      # faster
python mac_realtime.py --size 512 --t 35,45   # higher quality
python mac_realtime.py --bench --input some.png
```
Keys: `q` quit, `p` type a new prompt in the terminal. `--t` = t_index_list (lower = stronger transformation).
Your terminal app needs macOS camera permission.

## Benchmark (M2 Max 32GB, sd-turbo + TAESD, fp16)
| size | steps | ms/frame | FPS |
|---|---|---|---|
| 512 | 2 | 241 | 4.1 |
| 512 | 1 | 156 | 6.4 |
| 384 | 2 | 155 | 6.5 |
| 384 | 1 | 97 | 10.3 |
| 256 | 1 | 53 | 19.0 |
