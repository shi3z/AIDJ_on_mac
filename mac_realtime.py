"""StreamDiffusion realtime img2img on Apple Silicon (MPS).

Usage:
  python mac_realtime.py                       # webcam -> realtime img2img window
  python mac_realtime.py --bench               # benchmark only (no camera)
  python mac_realtime.py --prompt "anime girl, ..." --size 512

Keys in window: q = quit, p = type a new prompt in the terminal,
               l = re-learn MIDI knobs, 0 = reset all axes to center, f = toggle fullscreen,
               x = toggle the pixel-art display (with --pixel N), h = toggle the knob panel,
               k = knob setup (learn the 24 knobs again from the top-left; b = back one, n = skip)

AIDJ machine (default --mode machine): Launchpad Pro MK3 = 4 tracks x 16 step sequencer.
Each hit plays a drum voice and captures the meanings of that track's 6 knobs into the picture,
which keeps them until the track sounds again. See aidj_machine.py / aidj_axes.py.
--mode knobs: knobs steer the visual axes directly (no sequencer/audio).
Machine keys: space = play/stop, c = clear pattern, [ / ] = BPM -/+ 5,
              l = learn WORLDE controls (knobs, faders, mutes), n = skip a learn item
WORLDE: 24 knobs = 24 meanings composed together; each track's 3x2 pack = PITCH/RESONANCE/DELAY/HIGHPASS/REVERB/LOWPASS,
faders 1-4 = track volume, master fader, mute buttons 1-4.
"""
import argparse
import threading
import time

import cv2
import numpy as np
import torch
from diffusers import AutoencoderTiny, StableDiffusionPipeline
from PIL import Image

from aidj_axes import AxisSteering, MidiKnobs
from aidj_controls import Controller
from streamdiffusion import StreamDiffusion
from streamdiffusion.image_utils import postprocess_image


def build_stream(model_id, size, t_index_list, dtype, device):
    try:
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id, torch_dtype=dtype, variant="fp16", safety_checker=None)
    except (OSError, ValueError):  # model without an fp16 variant (e.g. IDKiro/sdxs-512-0.9)
        pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=dtype, safety_checker=None)
    pipe = pipe.to(device)
    stream = StreamDiffusion(
        pipe,
        t_index_list=t_index_list,
        torch_dtype=dtype,
        width=size,
        height=size,
        cfg_type="none",
        frame_buffer_size=1,
    )
    if isinstance(pipe.vae, AutoencoderTiny):  # the model ships its own tiny VAE (e.g. SDXS): keep it
        stream.vae = pipe.vae
    else:
        stream.vae = AutoencoderTiny.from_pretrained("madebyollin/taesd", torch_dtype=dtype).to(device)
        pipe.vae = stream.vae
    return stream


def to_bgr(tensor):
    img = postprocess_image(tensor, output_type="np")[0]  # HWC float 0..1
    return cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)


def frame_to_tensor(frame_bgr, device, dtype):
    """uint8 BGR HWC -> 1x3xHxW RGB in 0..1 on the GPU (StreamDiffusion normalizes to -1..1)."""
    t = torch.from_numpy(np.ascontiguousarray(frame_bgr[:, :, ::-1])).to(device)
    return t.permute(2, 0, 1).unsqueeze(0).to(dtype) / 255.0


def to_bgr_fast(tensor):
    """Decoder output (-1..1, 1x3xHxW on the GPU) -> uint8 BGR HWC, converted on the GPU."""
    x = ((tensor[0].clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8)
    return x[[2, 1, 0]].permute(1, 2, 0).contiguous().cpu().numpy()


class InferenceWorker(threading.Thread):
    """Runs camera read + generation in its own thread so the main thread's drawing and
    window handling overlap with the GPU work. The main thread picks up the newest result."""

    def __init__(self, cap, stream, steering, machine, args):
        super().__init__(daemon=True)
        self.cap, self.stream, self.steering, self.machine, self.args = cap, stream, steering, machine, args
        self.result = None  # (seq, frame_bgr, out_bgr, fps)
        self.new_prompt = None
        self.fps = 0.0
        self.ended = False
        self._stop = threading.Event()
        self._lock = threading.Lock()

    def latest(self):
        with self._lock:
            return self.result

    def stop(self):
        self._stop.set()

    @torch.no_grad()
    def run(self):
        args, seq, last = self.args, 0, time.perf_counter()
        fps_log_t = last
        while not self._stop.is_set():
            ok, frame = self.cap.read()
            if not ok and args.video and args.loop:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, frame = self.cap.read()
            if not ok:
                break
            frame = center_crop_square(frame, args.size)
            if not args.video:
                frame = cv2.flip(frame, 1)
            if self.new_prompt:
                print(f"prompt -> {self.new_prompt}", flush=True)
                self.steering.set_base_prompt(self.new_prompt)
                self.new_prompt = None
            if self.machine:
                self.steering.set_values(self.machine.visual_values())
            self.steering.apply()
            out = self.stream(frame_to_tensor(frame, self.stream.device, self.stream.dtype))
            out_bgr = to_bgr_fast(out)
            now = time.perf_counter()
            self.fps = 0.9 * self.fps + 0.1 * (1.0 / (now - last))
            last = now
            if now - fps_log_t > 10:
                print(f"[fps] {self.fps:.1f}", flush=True)
                fps_log_t = now
            seq += 1
            with self._lock:
                self.result = (seq, frame, out_bgr, self.fps)
        self.ended = True


def center_crop_square(frame, size):
    h, w = frame.shape[:2]
    s = min(h, w)
    y, x = (h - s) // 2, (w - s) // 2
    return cv2.resize(frame[y : y + s, x : x + s], (size, size))


def screen_size():
    """Logical size of the main display (points), e.g. 2056x1329 on a 16" MacBook Pro."""
    import subprocess
    try:
        out = subprocess.check_output(
            ["osascript", "-l", "JavaScript", "-e",
             'ObjC.import("AppKit"); var f=$.NSScreen.mainScreen.frame.size; f.width+"x"+f.height'],
            text=True).strip()
        w, h = (int(float(v)) for v in out.split("x"))
        return w, h
    except Exception:
        return 1920, 1080


STRIP_H = 34  # height of the bottom status strip in "stage" layout
PANEL_H = 128  # height of the knob panel above the strip (h toggles it)
TRACK_BGR = [(91, 86, 240), (74, 149, 245), (71, 197, 232), (232, 195, 76)]  # KICK SNARE HAT PERC
SOUND_SHORT = [["PIT", "RES"], ["DLY", "HPF"], ["REV", "LPF"]]  # pack [row][column in pack]


def draw_knob_panel(canvas, y0, W, st):
    """3 x 8 table of the WORLDE knobs: meaning, live value bar, sound role; 2 columns per track."""
    from aidj_axes import SEMANTIC_AXES
    from aidj_machine import TRACK_NAMES
    font = cv2.FONT_HERSHEY_SIMPLEX
    canvas[y0 : y0 + PANEL_H] = (24, 20, 20)
    cw = W // 8
    # header: one block per track (2 columns), flashes when the track sounds
    for tr in range(4):
        x = tr * 2 * cw
        lit = st["hit_age"][tr] < 0.15
        col = TRACK_BGR[tr]
        cv2.rectangle(canvas, (x + 2, y0 + 2), (x + 2 * cw - 2, y0 + 16), col if lit else (45, 40, 40), -1)
        last = st["last_step"][tr]
        txt = f"{TRACK_NAMES[tr]}" + ("  MUTE" if st["mute"][tr] else "") + (f"  last step {last + 1}" if last >= 0 else "")
        cv2.putText(canvas, txt, (x + 6, y0 + 13), font, 0.38, (20, 20, 20) if lit else col, 1, cv2.LINE_AA)
    # 3 rows x 8 knobs
    rh = (PANEL_H - 36) // 3
    for i in range(24):
        r, c = divmod(i, 8)
        x, y = c * cw, y0 + 20 + r * rh
        name, neg, _, pos, _ = SEMANTIC_AXES[i]
        v = st["knobs"][i] * 2 - 1
        col = TRACK_BGR[c // 2]
        if st.get("learn") == ("knob", i):  # knob setup: the knob to turn now, blinking
            on = int(time.perf_counter() * 4) % 2 == 0
            cv2.rectangle(canvas, (x + 1, y - 2), (x + cw - 2, y + rh - 4), (255, 120, 255) if on else (120, 40, 120), -1 if on else 2)
        cv2.putText(canvas, f"{neg}|{pos}", (x + 6, y + 11), font, 0.33, (190, 190, 190), 1, cv2.LINE_AA)
        snd = SOUND_SHORT[r][c % 2]
        cv2.putText(canvas, snd, (x + cw - 34, y + 11), font, 0.33, col, 1, cv2.LINE_AA)
        # bipolar bar: center line, fill to the value
        bx0, bx1, by = x + 6, x + cw - 50, y + 16
        mid = (bx0 + bx1) // 2
        cv2.rectangle(canvas, (bx0, by), (bx1, by + 6), (60, 55, 55), -1)
        end = int(mid + v * (bx1 - bx0) / 2)
        if abs(v) >= 0.04:
            cv2.rectangle(canvas, (min(mid, end), by), (max(mid, end), by + 6), col, -1)
        cv2.line(canvas, (mid, by - 1), (mid, by + 7), (150, 150, 150), 1)
        lab = (pos if v > 0 else neg) if abs(v) >= 0.04 else "-"
        cv2.putText(canvas, f"{v:+.2f}", (bx1 + 4, by + 6), font, 0.3, (220, 220, 220), 1, cv2.LINE_AA)
    # what the picture shows right now
    active = [f"{(SEMANTIC_AXES[i][3] if v > 0 else SEMANTIC_AXES[i][1])} {abs(v):.2f}"
              for i, v in enumerate(st["picture"]) if abs(v) >= 0.04]
    line = f"PICTURE = {st['src']}: " + (", ".join(active) if active else "base (all neutral)")
    cv2.putText(canvas, line[:200], (6, y0 + PANEL_H - 6), font, 0.38, (255, 200, 120), 1, cv2.LINE_AA)


def compose_stage(out_bgr, cam_bgr, W, H, items, fps, highlight=False, pixel=0, panel=None):
    """Generated image stretched over the whole screen except a thin strip at the bottom;
    status (camera thumbnail, FPS, knob values) lives in the strip so nothing covers the image.
    pixel > 0: pixel-art look - shrink to pixel x pixel dots and enlarge without smoothing."""
    canvas = np.zeros((H, W, 3), np.uint8)
    img_h = H - STRIP_H - (PANEL_H if panel is not None else 0)
    if pixel:
        dots = cv2.resize(out_bgr, (pixel, pixel), interpolation=cv2.INTER_AREA)
        canvas[:img_h] = cv2.resize(dots, (W, img_h), interpolation=cv2.INTER_NEAREST)
    else:
        canvas[:img_h] = cv2.resize(out_bgr, (W, img_h), interpolation=cv2.INTER_LINEAR)
    if panel is not None:
        draw_knob_panel(canvas, img_h, W, panel)
    th = STRIP_H - 4
    canvas[H - STRIP_H + 2 : H - 2, 2 : 2 + th] = cv2.resize(cam_bgr, (th, th))
    font, scale = cv2.FONT_HERSHEY_SIMPLEX, 0.4
    x0 = th + 10
    if isinstance(items, tuple):  # (row 1, row 2)
        rows = [[f"{fps:.1f} FPS"] + list(items[0]), list(items[1])]
    else:
        rows = [[f"{fps:.1f} FPS"] + items[:2], items[2:]]
    for r, row in enumerate(rows):
        x, y = x0, H - STRIP_H + 14 + r * 14
        for i, it in enumerate(row):
            color = (120, 220, 120) if (r == 0 and i == 0) else ((200, 200, 200) if r == 0 else
                     ((255, 120, 255) if highlight else (120, 220, 255)))
            w = cv2.getTextSize(it, font, scale, 1)[0][0]
            if x + w > W - 4:
                break
            cv2.putText(canvas, it, (x, y), font, scale, color, 1, cv2.LINE_AA)
            x += w + 16
    return canvas


def compose_split(cam_bgr, out_bgr, fps, label, hud, size):
    """Input | output side by side with overlay text (used for the README demo)."""
    view = np.hstack([cam_bgr, out_bgr])
    for text, org, scale, color in ((f"{fps:.1f} FPS", (10, 30), 0.9, (0, 255, 0)),
                                    (label, (10, view.shape[0] - 12), 0.55, (255, 255, 255))):
        cv2.putText(view, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 5)
        cv2.putText(view, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2)
    for i, line in enumerate(hud):
        org = (size + 8, 18 + i * 16)
        cv2.putText(view, line, org, cv2.FONT_HERSHEY_PLAIN, 0.95, (0, 0, 0), 4)
        cv2.putText(view, line, org, cv2.FONT_HERSHEY_PLAIN, 0.95, (255, 255, 0), 1)
    return view


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="stabilityai/sd-turbo")
    # defaults = the AIDJ setup: 256 px generation shown as 256x256 pixel art, ~15 FPS on an M2 Max
    ap.add_argument("--prompt", default="a character portrait, painting, highly detailed")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--t", default="32", help="t_index_list (img2img strength; lower = more change)")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--video", default=None, help="use a video file instead of the webcam")
    ap.add_argument("--record", default=None, help="save the side-by-side view to this .mp4")
    ap.add_argument("--no-window", action="store_true", help="do not open a window (for recording)")
    ap.add_argument("--loop", action="store_true", help="loop the --video file")
    ap.add_argument("--window-pos", default=None, help="x,y position of the window, e.g. 80,120")
    ap.add_argument("--layout", default="stage", choices=["stage", "split"],
                    help="stage: generated image fullscreen + small status strip; split: input | output")
    ap.add_argument("--screen", default=None, help="stage canvas size WxH (default: main display)")
    ap.add_argument("--windowed", action="store_true", help="stage layout in a window instead of fullscreen")
    ap.add_argument("--pixel", type=int, default=256,
                    help="pixel-art display: N x N dots enlarged without smoothing (0 = off, e.g. 96); x toggles")
    ap.add_argument("--midi", default="auto", help="MIDI input name (substring), 'auto' or 'off'")
    ap.add_argument("--axis-gain", type=float, default=1.5, help="strength of the knob axes")
    ap.add_argument("--mode", default="machine", choices=["machine", "knobs"],
                    help="machine: Launchpad sequencer + drum synth + knobs; knobs: knobs drive the axes directly")
    ap.add_argument("--bpm", type=float, default=120)
    ap.add_argument("--no-audio", action="store_true")
    ap.add_argument("--av-offset", type=float, default=0.15,
                    help="delay the drums by this many seconds so they line up with the picture")
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

    cap = cv2.VideoCapture(args.video if args.video else args.camera)
    if not cap.isOpened():
        raise SystemExit("Camera could not be opened (check macOS camera permission for your terminal).")

    steering = AxisSteering(stream, gain=args.axis_gain)
    steering.set_base_prompt(args.prompt)
    # Open the WORLDE port before the Launchpad ports (see aidj_machine.Launchpad.__init__).
    knobs = None
    if args.midi != "off" and args.mode == "machine":
        try:
            knobs = Controller(None, None if args.midi == "auto" else args.midi)
        except Exception as e:
            print(f"[midi] disabled: {e}")
    machine = None
    if args.mode == "machine":
        from aidj_machine import AIDJMachine
        try:
            machine = AIDJMachine(bpm=args.bpm, use_audio=not args.no_audio, av_offset=args.av_offset)
        except Exception as e:
            print(f"[machine] Launchpad not available ({e}); running without pads")
            machine = AIDJMachine(bpm=args.bpm, use_launchpad=False, use_audio=not args.no_audio,
                                  av_offset=args.av_offset)
        machine.start()
        if knobs is not None:
            knobs.target = machine
            knobs.start()
    if args.midi != "off" and args.mode == "knobs":
        try:
            knobs = MidiKnobs(steering, None if args.midi == "auto" else args.midi)
            knobs.start()
        except Exception as e:
            print(f"[midi] disabled: {e}")

    # Prompt editing from terminal without blocking the render loop

    def ask_prompt():
        worker.new_prompt = input("new prompt> ").strip() or None

    with torch.no_grad():
        for _ in range(len(t_index_list) + 1):
            ok, frame = cap.read()
            stream(frame_to_tensor(center_crop_square(frame, args.size), stream.device, stream.dtype))

    import platform, subprocess
    try:
        chip = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
    except Exception:
        chip = platform.machine()
    label = f"{chip} / PyTorch MPS / {args.size}px"
    win = "AIDJ" if args.layout == "stage" else "StreamDiffusion (MPS)"
    stage = args.layout == "stage"
    SW, SH = (int(v) for v in args.screen.split("x")) if args.screen else screen_size()
    fullscreen = stage and not args.windowed
    if not args.no_window:
        cv2.namedWindow(win, cv2.WINDOW_NORMAL if stage else cv2.WINDOW_AUTOSIZE)
        if args.window_pos:
            x, y = (int(v) for v in args.window_pos.split(","))
            cv2.moveWindow(win, x, y)
        if fullscreen:
            cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        elif stage:
            cv2.resizeWindow(win, SW // 2, SH // 2)

    worker = InferenceWorker(cap, stream, steering, machine, args)
    worker.start()
    asker = None
    writer = None
    shown = 0
    pixel_on = args.pixel > 0
    panel_on = True
    while True:
        res = worker.latest()
        if res is None or res[0] == shown:
            if worker.ended:
                break
            if args.no_window:
                time.sleep(0.002)
                continue
            key = cv2.waitKey(2) & 0xFF
        else:
            shown, frame, out_bgr, fps_ema = res
            if stage:
                if machine:
                    learn = knobs.prompt() if knobs is not None else None
                    items = (machine.status_items(), [learn] if learn else steering.status_items())
                else:
                    items = steering.status_items()
                view = compose_stage(out_bgr, frame, SW, SH, items, fps_ema,
                                     highlight=bool(machine and knobs and knobs.prompt()),
                                     pixel=args.pixel if pixel_on else 0,
                                     panel=dict(machine.panel_state(),
                                                learn=knobs.learn_target() if knobs is not None else None)
                                     if (machine and (panel_on or (knobs is not None and knobs.learning()))) else None)
            else:
                view = compose_split(frame, out_bgr, fps_ema, label, steering.hud_lines(), args.size)
            if args.record:
                if writer is None:
                    writer = cv2.VideoWriter(args.record, cv2.VideoWriter_fourcc(*"mp4v"), 10,
                                             (view.shape[1], view.shape[0]))
                writer.write(view)
            if args.no_window:
                continue
            cv2.imshow(win, view)
            key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("l") and knobs is not None:
            knobs.start_learn() if machine else knobs.reset_learn()
        if key == ord("n") and machine and knobs is not None:
            knobs.skip()
        if key == ord("k") and machine and knobs is not None:
            knobs.start_knob_learn()
        if key == ord("b") and machine and knobs is not None:
            knobs.back()
        if key == ord("f") and stage:
            fullscreen = not fullscreen
            cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN,
                                  cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL)
        if machine and key == ord(" "):
            machine.toggle_play()
        if machine and key == ord("c"):
            machine.clear_pattern()
        if machine and key in (ord("["), ord("]")):
            machine.bpm = max(40, min(240, machine.bpm + (5 if key == ord("]") else -5)))
        if key == ord("0"):
            steering.reset()
        if key == ord("h"):
            panel_on = not panel_on
        if key == ord("x") and args.pixel:
            pixel_on = not pixel_on
        if key == ord("p") and (asker is None or not asker.is_alive()):
            asker = threading.Thread(target=ask_prompt, daemon=True)
            asker.start()

    worker.stop()
    worker.join(timeout=2)
    if machine:
        machine.close()
    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
