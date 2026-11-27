"""AIDJ machine: Launchpad Pro MK3 step sequencer + WORLDE knobs + drum synth driving the visuals.

Pads (Launchpad Pro MK3, 8x8): 4 tracks x 16 steps. Rows from the top, two rows per track:
  rows 8-7 = track 1 (KICK), 6-5 = track 2 (SNARE), 4-3 = track 3 (HAT), 2-1 = track 4 (PERC);
  the upper row of a pair is steps 1-8, the lower row steps 9-16. Pressing a pad toggles the step.
Knobs (WORLDE, 3 rows x 8): 24 different meanings (aidj_axes.SEMANTIC_AXES), composed together.
  Two columns per track (cols 1-2 = track 1 ... 7-8 = track 4).
Pressing a pad turns the step on and records a snapshot of ALL 24 knobs into it. When the step
sounds, the picture becomes that whole snapshot (24 meanings) and holds until the next sounding
step. Same step on several tracks: the lowest track number's snapshot is shown.
Steps store only the picture. Sound follows the knobs in real time: each track's 2 columns x 3 rows
are its "pack" of 6 sound knobs, read at the moment the step sounds:
      left column      right column
  row 1  PITCH         RESONANCE      (+-1 octave / resonance of the high- and low-pass)
  row 2  DELAY         HIGHPASS       (dotted-8th echoes / high-pass)
  row 3  REVERB        LOWPASS
  Center = original pitch and open filters; delay and reverb are light there.
Turning a knob changes the picture right away (live state of all 24 knobs); the sound changes only
through recorded steps. The picture always shows the last event: a knob turn or a sounding step.
Faders 1-4 = track volumes, fader 5 = master. Mute buttons 1-4 mute the tracks (sound and picture).
WORLDE controls are learned on screen (aidj_controls.py).

Launchpad control follows pad_samply.py: firmware 385 has no programmer mode, so we use
DAW mode + Session layout; LEDs/SysEx go to the DAW port and presses are read from the DAW port
(not the MIDI port, which streams MIDI clock - see Launchpad.__init__).
"""
import collections
import functools
import json
import os
import threading
import time

import mido
import numpy as np
from scipy import signal

SR = 44100

# ---------------------------------------------------------------- drum synth
TRACK_NAMES = ["KICK", "SNARE", "HAT", "PERC"]
TRACK_COLORS = [5, 9, 13, 37]       # bright: red, orange, yellow, light blue
PLAYHEAD = 3                        # white, shown on the top button row (CC 91-98)


def _env(n, decay_s):
    return np.exp(-np.arange(n) / (decay_s * SR))


@functools.lru_cache(maxsize=512)
def _raw_drum(track, ratio=1.0):
    """Dry one-shot for a track; ratio = pitch multiplier (cached; don't modify the result)."""
    rng = np.random.default_rng(track)
    if track == 0:  # kick: sine with falling pitch
        n = int(0.6 * SR)
        f = (45 + 110 * _env(n, 0.035)) * ratio
        x = np.sin(2 * np.pi * np.cumsum(f) / SR) * _env(n, 0.28)
        x[: int(0.002 * SR)] += 0.4 * rng.standard_normal(int(0.002 * SR))
    elif track == 1:  # snare: tone + band-passed noise
        n = int(0.4 * SR)
        t = np.arange(n) / SR
        tone = np.sin(2 * np.pi * 185 * ratio * t) * _env(n, 0.06)
        b, a = signal.butter(2, [min(1500 * ratio, 15000), min(7000 * ratio, 19000)], "bandpass", fs=SR)
        noise = signal.lfilter(b, a, rng.standard_normal(n)) * _env(n, 0.12)
        x = 0.6 * tone + 1.4 * noise
    elif track == 2:  # closed hat: high-passed noise
        n = int(0.2 * SR)
        b, a = signal.butter(4, min(7000 * ratio, 19000), "highpass", fs=SR)
        x = signal.lfilter(b, a, rng.standard_normal(n)) * _env(n, 0.035) * 1.5
    else:  # perc: little FM blip / tom
        n = int(0.5 * SR)
        t = np.arange(n) / SR
        f = 330 * ratio * (1 + 0.5 * _env(n, 0.02))
        mod = np.sin(2 * np.pi * f * 1.5 * t) * 2.0 * _env(n, 0.08)
        x = np.sin(2 * np.pi * np.cumsum(f) / SR + mod) * _env(n, 0.18)
    return x / (np.abs(x).max() + 1e-9)


@functools.lru_cache(maxsize=1)
def _reverb_ir():
    n = int(1.8 * SR)
    rng = np.random.default_rng(7)
    ir = rng.standard_normal(n) * _env(n, 0.45)
    b, a = signal.butter(1, 6000, "lowpass", fs=SR)
    ir = signal.lfilter(b, a, ir)
    return ir / np.sqrt((ir ** 2).sum())


# Sound pack: each track's 2 knob columns x 3 rows. Left column top->bottom, then right column.
PACK_PARAMS = [["PITCH", "RESONANCE"], ["DELAY", "HIGHPASS"], ["REVERB", "LOWPASS"]]  # [row][col]


def _biquad(fc, q, kind):
    """RBJ resonant low-pass ("low") or high-pass ("high") biquad."""
    w0 = 2 * np.pi * min(fc, 0.45 * SR) / SR
    alpha = np.sin(w0) / (2 * q)
    c = np.cos(w0)
    if kind == "low":
        b = np.array([(1 - c) / 2, 1 - c, (1 - c) / 2])
    else:
        b = np.array([(1 + c) / 2, -(1 + c), (1 + c) / 2])
    a = np.array([1 + alpha, -2 * c, 1 - alpha])
    return b / a[0], a / a[0]


@functools.lru_cache(maxsize=256)
def render_voice(track, pitch_q, resonance_q, delay_q, highpass_q, reverb_q, lowpass_q, bpm):
    """Knob values quantized to 0..64 -> float32 mono buffer (cached).
    Curves are chosen so a knob at its center gives a mild, natural sound."""
    pitch, res, delay, hp, rev, lp = (v / 64 for v in (pitch_q, resonance_q, delay_q, highpass_q, reverb_q, lowpass_q))
    # pitch: one octave down (left) .. original (center) .. one octave up (right)
    x = _raw_drum(track, round(2 ** ((pitch - 0.5) * 2), 3)).copy()
    q = 0.707 * 16 ** (res ** 2)  # resonance Q 0.7 .. 11, applied to the high- and low-pass
    # Both filters are open at the knob's center, so an untouched pack sounds natural.
    # high-pass (resonant): off left of center; center -> right sweeps 20 Hz .. 5 kHz
    if hp > 0.52:
        b, a = _biquad(20 * 250 ** ((hp - 0.5) * 2), q, "high")
        x = signal.lfilter(b, a, x)
    # low-pass (resonant): off right of center; center -> left sweeps 20 kHz .. 200 Hz
    if lp < 0.48:
        b, a = _biquad(20000 * 0.01 ** ((0.5 - lp) * 2), q, "low")
        x = signal.lfilter(b, a, x)
    x = x / max(1.0, np.abs(x).max())  # resonance can boost a lot
    # delay: dotted 8th (3 steps) echoes; off at left, light at center, deep at right
    wet_d = 0.9 * delay ** 1.5
    if wet_d > 0.02:
        d = int(3 * 60 / bpm / 4 * SR)
        fb = 0.3 + 0.45 * delay
        n_rep = 6
        y = np.zeros(len(x) + d * n_rep)
        y[: len(x)] += x
        for k in range(1, n_rep + 1):
            y[k * d : k * d + len(x)] += x * wet_d * fb ** (k - 1)
        x = y
    # reverb: off at left, light at center, deep at right
    wet = rev ** 1.5
    if wet > 0.02:
        tail = signal.fftconvolve(x, _reverb_ir())
        y = np.zeros(len(tail))
        y[: len(x)] += x * (1 - wet * 0.4)
        y += tail * wet * 4.5  # the IR has unit energy spread over ~2 s, so it needs gain
        x = y / max(1.0, np.abs(y).max())
    return (x * 0.5).astype(np.float32)


class AudioEngine:
    """Mixes triggered one-shots in the sounddevice callback."""

    def __init__(self, blocksize=256):
        import sounddevice as sd

        self.pending = collections.deque()
        self.voices = []  # [buffer, pos]
        self.master = 1.0
        self.stream = sd.OutputStream(samplerate=SR, channels=2, blocksize=blocksize,
                                      dtype="float32", callback=self._cb, latency="low")
        self.stream.start()

    def play(self, buf, delay=0.0):
        """delay (s): played later, so the sound lines up with the picture, which lags by a frame or two."""
        if delay > 0:
            buf = np.concatenate([np.zeros(int(delay * SR), np.float32), buf])
        self.pending.append(buf)

    def _cb(self, out, frames, _time, _status):
        while self.pending:
            self.voices.append([self.pending.popleft(), 0])
        mix = np.zeros(frames, np.float32)
        alive = []
        for v in self.voices:
            buf, pos = v
            chunk = buf[pos : pos + frames]
            mix[: len(chunk)] += chunk
            v[1] = pos + frames
            if v[1] < len(buf):
                alive.append(v)
        self.voices = alive
        mix = np.tanh(mix * self.master)  # master volume + soft limiter
        out[:, 0] = mix
        out[:, 1] = mix

    def close(self):
        self.stream.stop()
        self.stream.close()


# ---------------------------------------------------------------- Launchpad
LP_DAW = "Launchpad Pro MK3 LPProMK3 DAW"
LP_MIDI = "Launchpad Pro MK3 LPProMK3 MIDI"
LP_HDR = [0x00, 0x20, 0x29, 0x02, 0x0E]
GRID = [r * 10 + c for r in range(1, 9) for c in range(1, 9)]
EDGE = list(range(91, 99)) + list(range(101, 109)) + list(range(1, 9)) + \
    [r * 10 for r in range(1, 9)] + [r * 10 + 9 for r in range(1, 9)]


def pad_to_step(note):
    """Launchpad note (row*10+col, row 1 = bottom) -> (track, step) or None."""
    row, col = divmod(note, 10)
    if not (1 <= row <= 8 and 1 <= col <= 8):
        return None
    from_top = 8 - row                 # 0..7
    track, half = divmod(from_top, 2)  # upper row of the pair = steps 0-7
    return track, half * 8 + (col - 1)


def step_to_pad(track, step):
    half, col = divmod(step, 8)
    return (8 - (track * 2 + half)) * 10 + col + 1


class Launchpad:
    def __init__(self):
        self.out = mido.open_output(LP_DAW)
        # Only the DAW port: presses arrive there in DAW mode, and the MIDI port streams MIDI clock
        # (~48 msg/s). With that traffic, opening/closing other ports can deadlock inside
        # CoreMIDI/rtmidi (the input callback waits for the GIL while we hold it in openPort).
        self.ins = [mido.open_input(LP_DAW)]
        self._last_press = {}
        self.out.send(mido.Message("sysex", data=LP_HDR + [0x10, 0x01]))              # DAW mode on
        time.sleep(0.2)
        self.out.send(mido.Message("sysex", data=LP_HDR + [0x00, 0x00, 0x00, 0x00]))  # Session layout
        time.sleep(0.2)
        self.clear()

    def clear(self):
        for n in GRID:
            self.out.send(mido.Message("note_on", note=n, velocity=0))
        for n in EDGE:
            self.out.send(mido.Message("control_change", control=n, value=0))

    def led(self, note, color):
        self.out.send(mido.Message("note_on", note=note, velocity=color))

    def presses(self):
        """Yield pad notes pressed since the last call. A press can arrive on both the DAW and
        MIDI ports; the same pad within 80 ms counts once, so a toggle never fires twice."""
        now = time.perf_counter()
        for port in self.ins:
            for m in port.iter_pending():
                if m.type == "note_on" and m.velocity > 0:
                    if now - self._last_press.get(m.note, -1) < 0.08:
                        continue
                    self._last_press[m.note] = now
                    print(f"[pad] {port.name.split()[-1]} note {m.note}", flush=True)
                    yield m.note

    def top_led(self, col, color):
        self.out.send(mido.Message("control_change", control=91 + col, value=color))

    def close(self):
        self.clear()
        self.out.send(mido.Message("sysex", data=LP_HDR + [0x10, 0x00]))  # DAW mode off
        for port in self.ins + [self.out]:
            port.close()


# ---------------------------------------------------------------- the machine
class AIDJMachine:
    """Sequencer + knobs + audio. The render loop reads visual_values() every frame."""

    N_TRACKS, N_STEPS = 4, 16
    PATTERN_FILE = "aidj_pattern.json"

    def _save_pattern(self):
        with open(self.PATTERN_FILE, "w") as f:
            json.dump({"pattern": self.pattern.astype(int).tolist(),
                       "knobs": self.step_knobs.round(4).tolist()}, f)

    def __init__(self, bpm=120, use_launchpad=True, use_audio=True, av_offset=0.15):
        self.bpm = bpm
        self.av_offset = av_offset  # audio delay (s) to match the picture's latency
        self.pattern = np.zeros((self.N_TRACKS, self.N_STEPS), bool)
        # snapshot of ALL 24 knobs taken when each step's pad was pressed
        self.step_knobs = np.full((self.N_TRACKS, self.N_STEPS, 24), 0.5)
        if os.path.exists(self.PATTERN_FILE):  # the pattern survives restarts
            data = json.load(open(self.PATTERN_FILE))
            if isinstance(data, dict):
                self.pattern[:] = np.array(data["pattern"], bool)
                k = np.array(data["knobs"])
                if k.shape == self.step_knobs.shape:
                    self.step_knobs[:] = k
                else:  # older file: only the track's own 2 columns were recorded
                    for tr in range(self.N_TRACKS):
                        self.step_knobs[tr].reshape(self.N_STEPS, 3, 8)[:, :, 2 * tr : 2 * tr + 2] = k[tr]
            else:  # older file: pattern only
                self.pattern[:] = np.array(data, bool)
        self.knob_raw = np.full(24, 0.5)           # 3 rows x 8 knobs, 0..1, center = neutral
        self.volume = np.full(self.N_TRACKS, 0.8)  # faders 1-4
        self.mute = np.zeros(self.N_TRACKS, bool)  # mute buttons 1-4
        self.step = -1
        self.playing = True
        self._lock = threading.Lock()
        self._hit_t = np.full(self.N_TRACKS, -1e9)  # last time each track sounded
        self.latched = np.zeros(24)  # picture: live knobs or the snapshot of the last sounding step
        self.picture_src = "BASE"
        self.last_step = np.full(self.N_TRACKS, -1)
        self.audio = AudioEngine() if use_audio else None
        self.lp = Launchpad() if use_launchpad else None
        for tr in range(self.N_TRACKS):  # warm the voice cache for the default knob position
            render_voice(tr, 32, 32, 32, 32, 32, 32, int(bpm))
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        if self.lp:
            self._draw_all()

    # -- WORLDE controls (called from the MIDI thread)
    def set_knob(self, idx, raw01):
        """idx 0..23 row-major (3 rows x 8 cols); columns 2k, 2k+1 belong to track k."""
        with self._lock:
            self.knob_raw[idx] = raw01
            self.latched = self.knob_raw * 2 - 1  # live: turning a knob shows its state right away
            self.picture_src = "LIVE KNOBS"

    def set_fader(self, idx, raw01):
        """Faders 0-3 = track volumes, 4 = master volume."""
        if idx < self.N_TRACKS:
            self.volume[idx] = raw01
        elif self.audio and idx == self.N_TRACKS:
            self.audio.master = raw01 * 1.25

    def set_mute(self, idx, on):
        if idx < self.N_TRACKS:
            self.mute[idx] = on
            print(f"[mute] {TRACK_NAMES[idx]} {'MUTED' if on else 'on'}", flush=True)

    def toggle_mute(self, idx):
        if idx < self.N_TRACKS:
            self.set_mute(idx, not self.mute[idx])

    def visual_values(self):
        """24 axis values for the picture = the state of the last event:
        - turning a knob: the live state of all 24 knobs (shown right away)
        - a step sounding: the 24-knob snapshot recorded when its pad was pressed."""
        with self._lock:
            return self.latched.tolist()

    def panel_state(self):
        """Everything the on-screen knob panel shows."""
        with self._lock:
            return {"knobs": self.knob_raw.copy(), "picture": self.latched.copy(), "src": self.picture_src,
                    "hit_age": time.perf_counter() - self._hit_t, "last_step": self.last_step.copy(),
                    "mute": self.mute.copy(), "volume": self.volume.copy()}

    def status_items(self):
        """(row 1, row 2) for the status strip."""
        now = time.perf_counter()
        with self._lock:
            row1 = [f"BPM {self.bpm:.0f}", f"STEP {self.step + 1:>2}/16" + ("" if self.playing else " STOP")]
            for tr in range(self.N_TRACKS):
                hit = "*" if now - self._hit_t[tr] < 0.15 else " "
                vol = "MUTE" if self.mute[tr] else f"VOL{self.volume[tr]:.2f}"
                row1.append(f"{hit}{TRACK_NAMES[tr]} {vol}")
            if self.audio:
                row1.append(f"MASTER {self.audio.master:.2f}")
        return row1

    def toggle_play(self):
        self.playing = not self.playing

    def clear_pattern(self):
        with self._lock:
            self.pattern[:] = False
        self._save_pattern()
        if self.lp:
            self._draw_all()

    # -- sequencer
    def start(self):
        self._thread.start()

    def toggle_step(self, tr, st):
        """Pad press: step on/off. Turning a step on records a snapshot of all 24 knobs into it."""
        with self._lock:
            self.pattern[tr, st] = not self.pattern[tr, st]
            on = self.pattern[tr, st]
            if on:
                self.step_knobs[tr, st] = self.knob_raw
        if on:
            print(f"[rec] {TRACK_NAMES[tr]} step {st + 1}: {self._describe(self.step_knobs[tr, st])}", flush=True)
        self._save_pattern()
        if self.lp:
            self._draw_cell(tr, st)

    @staticmethod
    def _describe(snap):
        from aidj_axes import SEMANTIC_AXES
        out = []
        for i, raw in enumerate(snap):
            v = raw * 2 - 1
            if abs(v) >= 0.04:
                ax = SEMANTIC_AXES[i]
                out.append(f"{ax[3] if v > 0 else ax[1]} {abs(v):.2f}")
        return ", ".join(out) or "all neutral (knobs at center)"

    def _tick(self):
        with self._lock:
            self.step = (self.step + 1) % self.N_STEPS
            hits = [tr for tr in range(self.N_TRACKS) if self.pattern[tr, self.step] and not self.mute[tr]]
            rec = {tr: self.step_knobs[tr, self.step].copy() for tr in hits}  # 24-knob snapshots (picture)
            live = self.knob_raw.copy()  # sound follows the knobs in real time
        if not hits:
            return
        now = time.perf_counter()
        for tr in hits:
            # sound: the track's pack (its 2 columns x 3 rows) as the knobs are NOW, see PACK_PARAMS
            pk = np.round(live.reshape(3, 8)[:, 2 * tr : 2 * tr + 2] * 64).astype(int)
            if self.audio and self.volume[tr] > 0.001:
                buf = render_voice(tr, pk[0, 0], pk[0, 1], pk[1, 0], pk[1, 1], pk[2, 0], pk[2, 1],
                                   int(round(self.bpm)))
                # squared = more natural fader taper
                self.audio.play(buf * np.float32(self.volume[tr] ** 2), delay=self.av_offset)
        with self._lock:
            # the picture becomes the whole snapshot of the step that sounded; if several tracks
            # sound on the same step, the lowest track number wins (KICK > SNARE > HAT > PERC)
            self.latched = rec[hits[0]] * 2 - 1
            self.picture_src = f"{TRACK_NAMES[hits[0]]} STEP {self.step + 1} (recorded)"
            self._hit_t[hits] = now
            self.last_step[hits] = self.step

    def _run(self):
        next_t = time.perf_counter()
        prev = None
        while not self._stop.is_set():
            if self.lp:
                for note in self.lp.presses():
                    ts = pad_to_step(note)
                    if ts:
                        self.toggle_step(*ts)
            now = time.perf_counter()
            if now >= next_t:
                next_t += 60.0 / self.bpm / 4  # 16th notes
                if next_t < now:  # fell behind (e.g. stopped) - resync
                    next_t = now + 60.0 / self.bpm / 4
                if self.playing:
                    self._tick()
                    if self.lp:
                        # playhead on the top button row; pads only ever show on/off
                        if prev is not None and prev % 8 != self.step % 8:
                            self.lp.top_led(prev % 8, 0)
                        self.lp.top_led(self.step % 8, PLAYHEAD)
                        prev = self.step
            time.sleep(0.001)

    # -- LEDs: a pad is lit in its track color when the step is ON, unlit when OFF
    def _draw_cell(self, tr, st):
        self.lp.led(step_to_pad(tr, st), TRACK_COLORS[tr] if self.pattern[tr, st] else 0)

    def _draw_all(self):
        for tr in range(self.N_TRACKS):
            for st in range(self.N_STEPS):
                self._draw_cell(tr, st)

    def close(self):
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=1)
        if self.lp:
            self.lp.close()
        if self.audio:
            self.audio.close()
