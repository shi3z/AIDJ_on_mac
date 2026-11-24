"""StreamDiffusion realtime img2img on Apple Silicon (MPS).

Usage:
  python mac_realtime.py                       # webcam -> realtime img2img window
  python mac_realtime.py --bench               # benchmark only (no camera)
  python mac_realtime.py --prompt "anime girl, ..." --size 512

Keys in window: q = quit, p = type a new prompt in the terminal
"""
import argparse
import threading
import time

import cv2
import numpy as np
import torch
from diffusers import AutoencoderTiny, StableDiffusionPipeline
from PIL import Image

from streamdiffusion import StreamDiffusion
from streamdiffusion.image_utils import postprocess_image


def build_stream(model_id, size, t_index_list, dtype, device):
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id, torch_dtype=dtype, variant="fp16", safety_checker=None
    ).to(device)
    stream = StreamDiffusion(
        pipe,
        t_index_list=t_index_list,
        torch_dtype=dtype,
        width=size,
        height=size,
        cfg_type="none",
        frame_buffer_size=1,
    )
    stream.vae = AutoencoderTiny.from_pretrained("madebyollin/taesd", torch_dtype=dtype).to(device)
    pipe.vae = stream.vae
    return stream


def to_bgr(tensor):
    img = postprocess_image(tensor, output_type="np")[0]  # HWC float 0..1
    return cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)


def center_crop_square(frame, size):
    h, w = frame.shape[:2]
    s = min(h, w)
    y, x = (h - s) // 2, (w - s) // 2
    return cv2.resize(frame[y : y + s, x : x + s], (size, size))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="stabilityai/sd-turbo")
    ap.add_argument("--prompt", default="oil painting portrait, vivid colors, masterpiece, highly detailed")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--t", default="35,45", help="t_index_list (img2img strength; lower = more change)")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--input", default=None, help="input image for --bench")
    args = ap.parse_args()

    device = torch.device("mps")
    dtype = torch.float16
    t_index_list = [int(x) for x in args.t.split(",")]

    stream = build_stream(args.model, args.size, t_index_list, dtype, device)
    stream.prepare(prompt=args.prompt, num_inference_steps=50, guidance_scale=0.0)

    if args.bench:
        img = (Image.open(args.input).convert("RGB").resize((args.size, args.size))
               if args.input else Image.new("RGB", (args.size, args.size), (128, 100, 80)))
        for _ in range(len(t_index_list) + 3):  # warmup (fill the denoising batch)
            stream(img)
        n = 30
        t0 = time.perf_counter()
        for _ in range(n):
            out = stream(img)
        torch.mps.synchronize()
        dt = (time.perf_counter() - t0) / n
        print(f"[bench] {args.size}x{args.size}, steps={len(t_index_list)}: {dt*1000:.1f} ms/frame = {1/dt:.1f} FPS")
        Image.fromarray(cv2.cvtColor(to_bgr(out), cv2.COLOR_BGR2RGB)).save(f"bench_out_{args.size}_{len(t_index_list)}.png")
        return

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit("Camera could not be opened (check macOS camera permission for your terminal).")

    # Prompt editing from terminal without blocking the render loop
    new_prompt = {"v": None}

    def ask_prompt():
        new_prompt["v"] = input("new prompt> ").strip() or None

    for _ in range(len(t_index_list) + 1):
        ok, frame = cap.read()
        stream(Image.fromarray(cv2.cvtColor(center_crop_square(frame, args.size), cv2.COLOR_BGR2RGB)))

    fps_ema, last = 0.0, time.perf_counter()
    asker = None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(center_crop_square(frame, args.size), 1)
        out = stream(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        out_bgr = to_bgr(out)

        now = time.perf_counter()
        fps_ema = 0.9 * fps_ema + 0.1 * (1.0 / (now - last))
        last = now
        view = np.hstack([frame, out_bgr])
        cv2.putText(view, f"{fps_ema:.1f} FPS", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.imshow("StreamDiffusion (MPS)", view)

        if new_prompt["v"]:
            print(f"prompt -> {new_prompt['v']}")
            stream.update_prompt(new_prompt["v"])
            new_prompt["v"] = None

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("p") and (asker is None or not asker.is_alive()):
            asker = threading.Thread(target=ask_prompt, daemon=True)
            asker.start()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
