"""WORLDE controller: learn knobs, faders and mute buttons by following on-screen prompts.

Setup (k or l): 24 knobs (row 1 left->right, row 2, row 3), faders for tracks 1-4, master fader,
mute buttons for tracks 1-4; everything is learned again. The knob to turn is highlighted in the on-screen panel. A control is taken only once it
has clearly moved (value changed by 10+), so a bumped or jittering control isn't learned by mistake,
and a control already learned for something else is never taken. n skips an item, b goes back one.
Mute buttons may send CC or notes; press-only, momentary (press/release) and latching (on/off on
alternate presses) buttons all toggle the mute once per press.
The map is saved to worlde_map.json.
"""
import json
import os
import threading
import time

import mido

from aidj_axes import SEMANTIC_AXES

TRACKS = ["KICK", "SNARE", "HAT", "PERC"]
ITEMS = (
    [("knob", i, f"knob row{i // 8 + 1} #{i % 8 + 1} ({SEMANTIC_AXES[i][1]}<->{SEMANTIC_AXES[i][3]})") for i in range(24)]
    + [("fader", i, f"fader for {TRACKS[i]}") for i in range(4)]
    + [("fader", 4, "MASTER fader (the last one)")]
    + [("mute", i, f"MUTE button for {TRACKS[i]}") for i in range(4)]
)
KNOB_ITEMS = ITEMS[:24]
MOVE_THRESHOLD = 10  # a knob/fader is learned only after its value moved this much


def _key(msg):
    if msg.type == "control_change":
        return ("cc", msg.channel, msg.control), msg.value
    if msg.type in ("note_on", "note_off"):
        return ("note", msg.channel, msg.note), (msg.velocity if msg.type == "note_on" else 0)
    return None, None


class Controller(threading.Thread):
    def __init__(self, target, port_hint=None, map_path="worlde_map.json"):
        """target: object with set_knob(i, v01), set_fader(i, v01), set_mute(i, bool), toggle_mute(i)."""
        super().__init__(daemon=True)
        names = mido.get_input_names()
        if not names:
            raise RuntimeError("No MIDI input found")
        if port_hint:
            name = next((n for n in names if port_hint.lower() in n.lower()), names[0])
        else:
            name = next((n for n in names if "launchpad" not in n.lower()), names[0])
        self.port = mido.open_input(name)
        try:  # for mute-button lamps
            self.out = mido.open_output(name)
        except Exception:
            self.out = None
        self.target = target
        self.map_path = map_path
        self.map = {}  # key tuple -> (kind, index, mode)
        self.items = ITEMS  # what the current learn session goes through
        self.learn_pos = len(ITEMS)
        self._cand = {}  # key -> first value seen while learning (movement check)
        self._continuous = set()  # keys that sent a value other than 0/127: knobs/faders, not buttons
        self._lock = threading.Lock()
        self._pending_button = None  # (key, idx, t0, saw_zero)
        self._last_button = {}  # mute key -> (last value, time)
        if os.path.exists(map_path):
            with open(map_path) as f:
                data = json.load(f)
            for e in data["controls"]:
                self.map[tuple(e["key"])] = (e["kind"], e["index"], e.get("mode"))
            self.items = KNOB_ITEMS if data.get("learn_mode") == "knobs" else ITEMS
            self.learn_pos = data.get("learn_pos", len(self.items))
            print(f"[ctrl] {name}: loaded {len(self.map)} controls from {map_path}", flush=True)
            if self.learning():  # resume an unfinished learn session
                print(f"[ctrl] learn: {self.prompt()}", flush=True)
        else:
            self.start_learn()

    # ---- learning
    def start_learn(self):
        """Full learn: knobs, faders, mutes."""
        with self._lock:
            self.map = {}
            self.items, self.learn_pos, self._cand = ITEMS, 0, {}
            self._save()
        print(f"[ctrl] learn: {self.prompt()}", flush=True)

    def start_knob_learn(self):
        """Setup from the top-left knob, then the faders and mute buttons. Everything is learned
        again: a knob that an older learn had taken as a fader or mute is free to be a knob now."""
        self.start_learn()

    def skip(self):
        with self._lock:
            if self.learning():
                print(f"[ctrl] skipped: {self.items[self.learn_pos][2]}", flush=True)
                self._advance()

    def back(self):
        """Go back one item and forget what was learned for it."""
        with self._lock:
            if self.learn_pos == 0:
                return
            self.learn_pos -= 1
            kind, idx, label = self.items[self.learn_pos]
            self.map = {k: v for k, v in self.map.items() if not (v[0] == kind and v[1] == idx)}
            self._cand = {}
            self._save()
        print(f"[ctrl] back: {self.prompt()}", flush=True)

    def learning(self):
        return self.learn_pos < len(self.items)

    def learn_target(self):
        """(kind, index) the user should move now, or None."""
        if not self.learning():
            return None
        kind, idx, _ = self.items[self.learn_pos]
        return kind, idx

    def prompt(self):
        if not self.learning():
            return None
        mode = "SETUP"
        return (f"{mode} {self.learn_pos + 1}/{len(self.items)}: turn {self.items[self.learn_pos][2]} "
                f"(highlighted)  n = skip, b = back")

    def _advance(self):
        self.learn_pos += 1
        self._cand = {}
        self._save()  # save after every item so a restart resumes where it stopped
        if self.learning():
            print(f"[ctrl] learn: {self.prompt()}", flush=True)
        else:
            print(f"[ctrl] done -> {self.map_path}", flush=True)

    def _save(self):
        with open(self.map_path, "w") as f:
            json.dump({"learn_pos": self.learn_pos,
                       "learn_mode": "knobs" if self.items is KNOB_ITEMS else "all",
                       "controls": [{"key": list(k), "kind": v[0], "index": v[1], "mode": v[2]}
                                    for k, v in self.map.items()]}, f, indent=1)

    # ---- main loop
    def run(self):
        while True:
            for msg in self.port.iter_pending():
                self._handle(msg)
            self._finish_button_learn()
            time.sleep(0.002)

    def _finish_button_learn(self):
        pb = self._pending_button
        if pb and time.perf_counter() - pb[2] > 0.8:  # decide momentary vs latching
            key, idx, _, saw_zero = pb
            with self._lock:
                self.map[key] = ("mute", idx, "momentary" if saw_zero else "latching")
                self._pending_button = None
                print(f"[ctrl] {key} -> mute {TRACKS[idx]} ({self.map[key][2]})", flush=True)
                self._advance()

    def _handle(self, msg):
        key, value = _key(msg)
        if key is None:
            return
        with self._lock:
            pb = self._pending_button
            if pb and key == pb[0]:
                if value == 0:
                    self._pending_button = (pb[0], pb[1], pb[2], True)
                return
            if 0 < value < 127:
                self._continuous.add(key)
            if self.learning() and key not in self.map and not pb:
                kind, idx, label = self.items[self.learn_pos]
                if kind == "mute":
                    if key in self._continuous:  # a knob or fader, not a button
                        return
                    if value > 0:
                        self._pending_button = (key, idx, time.perf_counter(), False)
                    return
                first = self._cand.setdefault(key, value)
                if abs(value - first) < MOVE_THRESHOLD:  # not clearly turned yet
                    return
                self.map[key] = (kind, idx, None)
                print(f"[ctrl] {key} -> {label}", flush=True)
                self._advance()
            entry = self.map.get(key)
        if entry is None:
            return
        kind, idx, mode = entry
        if kind == "knob":
            self.target.set_knob(idx, value / 127)
        elif kind == "fader":
            self.target.set_fader(idx, value / 127)
        elif kind == "mute":
            # Works for every button type without trusting the learned "mode" (the same hardware
            # was classified differently depending on how long it was held while learning):
            #  - a press (value > 0) toggles the mute; press-only buttons send just this
            #  - a 0 shortly after a press is the release of a momentary button: ignored
            #  - a 0 long after the last press is a latching button's second press: toggles
            now = time.perf_counter()
            last_v, last_t = self._last_button.get(key, (0, -1e9))
            self._last_button[key] = (value, now)
            if value > 0 or (last_v > 0 and now - last_t > 0.6):
                self.target.toggle_mute(idx)
            self._mute_lamp(key, bool(self.target.mute[idx]))

    def _mute_lamp(self, key, on):
        """Keep the button lamp lit while the track is muted. The WORLDE switches its own lamp
        off on release, so send the state now and again a moment later."""
        if self.out is None:
            return
        kind, ch, num = key
        msg = (mido.Message("control_change", channel=ch, control=num, value=127 if on else 0) if kind == "cc"
               else mido.Message("note_on", channel=ch, note=num, velocity=127 if on else 0))
        self.out.send(msg)
        threading.Timer(0.03, self.out.send, args=(msg,)).start()
