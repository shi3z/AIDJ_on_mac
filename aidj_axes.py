"""AIDJ: steer the prompt embedding with semantic axes controlled by MIDI knobs.

Each semantic axis has two poles (e.g. machine <-> nature). For a base prompt P:
  E0  = CLIP(P)
  E+  = CLIP(P + ", " + positive pole words)
  E-  = CLIP(P + ", " + negative pole words)
A knob value v in [-1, 1] (center = 0) adds  v * gain * (E+ - E0)  for v > 0,
or  |v| * gain * (E- - E0)  for v < 0. All axes are summed on top of E0.

Layout: 3 rows x 8 knobs, a different meaning on every knob; all of them are composed together.
"""
import json
import os
import threading

import torch

# 3 rows x 8 knobs, one meaning per knob (row-major: index = row * 8 + col).
# Axis: (name, neg label, neg words, pos label, pos words)
SEMANTIC_AXES = [
    # Row 1
    ("animal-machine", "ANIMAL", "animal, furry creature, wildlife, fur and feathers",
     "MACHINE", "machine, mechanical robot, metal gears, cyborg, chrome"),
    ("human-plant", "HUMAN", "human, person, human face and body, people",
     "PLANT", "plant, leaves, flowers, vines, botanical, moss"),
    ("nonliving-living", "NONLIVING", "inanimate object, lifeless, statue, mannequin, still object",
     "LIVING", "living creature, alive, breathing organism, biological life"),
    ("west-east", "WEST", "western european style, gothic, renaissance, baroque",
     "EAST", "east asian style, japanese ukiyo-e, chinese ink painting, temple, kimono"),
    ("cute-monster", "CUTE", "cute, adorable, kawaii, chibi, round soft shapes",
     "MONSTER", "monstrous, grotesque creature, horns, fangs, terrifying beast"),
    ("young-old", "YOUNG", "young, child, youthful, fresh",
     "OLD", "elderly, ancient, wrinkled, weathered, aged"),
    ("holy-demon", "HOLY", "holy, angelic, divine light, halo, sacred, white wings",
     "DEMON", "demonic, infernal, dark occult, hellfire, sinister"),
    ("flesh-crystal", "ORGANIC", "organic flesh, biological, veins, soft tissue",
     "CRYSTAL", "crystal, mineral, gemstone, glass, faceted stone"),
    # Row 2
    ("day-night", "DAY", "bright daylight, sunny, blue sky, midday",
     "NIGHT", "night, darkness, moonlight, starry sky, city lights"),
    ("summer-winter", "SUMMER", "summer, hot sun, tropical, beach",
     "WINTER", "winter, snow, frost, cold, snowfall"),
    ("calm-chaos", "CALM", "calm, serene, peaceful, still, tranquil",
     "CHAOS", "chaotic, explosive, storm, debris, motion blur, frenzy"),
    ("joy-gloom", "JOY", "joyful, happy, cheerful, festive, smiling",
     "GLOOM", "gloomy, melancholic, sad, bleak, despair"),
    ("nature-city", "NATURE", "wild nature, forest, mountains, wilderness",
     "CITY", "urban city, skyscrapers, streets, concrete, traffic"),
    ("sea-space", "SEA", "underwater, deep ocean, coral reef, bubbles, fish",
     "SPACE", "outer space, galaxy, nebula, planets, astronaut"),
    ("fire-ice", "FIRE", "fire, flames, lava, embers, burning",
     "ICE", "ice, frozen, icicles, glacier, frost crystals"),
    ("real-surreal", "REAL", "realistic everyday scene, documentary, ordinary",
     "SURREAL", "surreal, dreamlike, salvador dali, impossible geometry, melting"),
    # Row 3
    ("past-future", "PAST", "ancient past, antique, historical, medieval, old weathered",
     "FUTURE", "far future, futuristic, sci-fi, sleek, holographic, year 3000"),
    ("photo-painting", "PHOTO", "photograph, dslr photo, 35mm, photorealistic",
     "PAINTING", "oil painting, brush strokes, canvas, impasto"),
    ("real-anime", "REALISTIC", "realistic, lifelike, detailed skin texture",
     "ANIME", "anime, manga, cel shading, japanese animation"),
    ("mono-vivid", "MONO", "black and white, monochrome, grayscale",
     "VIVID", "vivid saturated colors, colorful, rainbow"),
    ("minimal-ornate", "MINIMAL", "minimalist, simple, flat, clean, negative space",
     "ORNATE", "ornate, intricate, highly detailed ornament, filigree, maximalist"),
    ("sketch-3d", "SKETCH", "pencil sketch, line drawing, graphite, hand drawn",
     "3D", "3d render, octane render, cgi, unreal engine"),
    ("pastel-neon", "PASTEL", "soft pastel colors, gentle, muted",
     "NEON", "neon glow, fluorescent, glowing lights, synthwave"),
    ("micro-giant", "MICRO", "microscopic, tiny, macro lens close-up, miniature world",
     "GIANT", "gigantic, colossal, towering, epic scale, titan"),
]
N_KNOBS = len(SEMANTIC_AXES)  # 24


class AxisSteering:
    """Holds the 3 axis values (set from 24 knobs) and applies them to the stream once per frame."""

    MAX_OFFSET_RATIO = 1.1  # |offset| / |E0|; one axis at full with gain 1.5 is ~1.07

    def __init__(self, stream, gain=1.5, axes=SEMANTIC_AXES):
        self.stream = stream
        self.axes = axes
        self.values = [0.0] * len(axes)  # -1..1 per semantic axis
        self.gain = gain
        self._lock = threading.Lock()
        self._dirty_embed = True
        self.base_prompt = None

    # ---- knob inputs (called from the MIDI thread) ----
    def set_knob(self, idx, raw01):
        """idx: 0..23 knob index (row-major), raw01: 0..1 knob position."""
        self.set_value(idx, raw01 * 2 - 1)

    def set_values(self, vals):
        """Set all axis values at once (render thread, from the machine)."""
        new = [0.0 if abs(v) < 0.04 else max(-1.5, min(1.5, v)) for v in vals]
        with self._lock:
            if new != self.values:  # recompute the embedding only when something changed
                self.values = new
                self._dirty_embed = True

    def set_value(self, i, v):
        with self._lock:
            v = max(-1.0, min(1.0, v))
            if abs(v) < 0.04:  # dead zone around the knob's center
                v = 0.0
            if v != self.values[i]:
                self.values[i] = v
                self._dirty_embed = True

    def reset(self):
        for i in range(len(self.axes)):
            self.set_value(i, 0.0)

    # ---- render thread ----
    @torch.no_grad()
    def _encode(self, texts):
        return self.stream.pipe.encode_prompt(
            prompt=texts, device=self.stream.device, num_images_per_prompt=1,
            do_classifier_free_guidance=False,
        )[0]

    @torch.no_grad()
    def set_base_prompt(self, prompt):
        texts = [prompt]
        for _, _, neg, _, pos in self.axes:
            texts += [f"{prompt}, {neg}", f"{prompt}, {pos}"]
        e = self._encode(texts)  # one batched CLIP call for all prompts
        self.e0 = e[0:1]
        d_neg = e[1::2] - self.e0  # (n_axes, 77, 768)
        d_pos = e[2::2] - self.e0
        self.d_neg, self.d_pos = d_neg, d_pos
        self.base_prompt = prompt
        self._dirty_embed = True

    @torch.no_grad()
    def apply(self):
        """Call once per frame from the render thread; updates the stream if knobs moved."""
        if not self._dirty_embed:
            return
        with self._lock:
            vals = torch.tensor(self.values, device=self.e0.device, dtype=self.e0.dtype)
            gain = self.gain
            self._dirty_embed = False
        # Each pole offset = common part c (what appending *any* words does) + axis-specific part.
        # The axis-specific parts add up; the common part is averaged, so turning many knobs
        # mixes their meanings instead of stacking the shared "extra words" shift N times.
        # With a single knob this equals the plain  |v| * gain * (E± - E0).
        a = vals.abs()
        d = torch.where((vals >= 0).view(-1, 1, 1), self.d_pos, self.d_neg)
        c = (self.d_pos + self.d_neg) / 2
        spec = d - c
        w = (a * gain).view(-1, 1, 1)
        common = (w * c).sum(0, keepdim=True) / a.sum().clamp(min=1.0)
        off = (w * spec).sum(0, keepdim=True) + common
        # Keep the total push bounded so many knobs at once don't blow the embedding up into noise.
        ratio = off.float().norm() / self.e0.float().norm()
        if ratio > self.MAX_OFFSET_RATIO:
            off = off * (self.MAX_OFFSET_RATIO / ratio).to(off.dtype)
        e = self.e0 + off
        self.stream.prompt_embeds = e.repeat(self.stream.batch_size, 1, 1)

    def hud_lines(self):
        """One bar per axis (split layout)."""
        out = []
        for (_, nl, _, pl, _), v in zip(self.axes, self.values):
            n = 10
            bar = ["-"] * (n + 1)
            bar[n // 2] = "+"
            bar[int(round((v + 1) / 2 * n))] = "#"
            out.append(f"{nl:>9} {''.join(bar)} {pl:<8} {v:+.2f}")
        return out

    def status_items(self):
        """Compact status for the bottom strip: engine values + off-center axes as pole labels."""
        items = []
        for (_, nl, _, pl, _), v in zip(self.axes, self.values):
            if v:
                items.append(f"{pl if v > 0 else nl} {abs(v):.2f}")
        return items

    @staticmethod
    def knob_name(idx):
        row, col = divmod(idx, 8)
        return f"row{row + 1}-knob{col + 1} ({SEMANTIC_AXES[idx][0]})"


class MidiKnobs(threading.Thread):
    """Reads CC messages; maps distinct CCs to knobs in the order they are first touched
    (MIDI learn: row 1 left->right, then row 2, then row 3); knob index // 8 = axis, saved to map_path."""

    def __init__(self, steering, port_hint=None, map_path="midi_map.json"):
        """steering: any object with set_knob(idx 0..23, value 0..1) (AxisSteering or AIDJMachine)."""
        super().__init__(daemon=True)
        import mido

        names = mido.get_input_names()
        if not names:
            raise RuntimeError("No MIDI input found")
        if port_hint:
            name = next((n for n in names if port_hint.lower() in n.lower()), names[0])
        else:  # auto: first input that is not the Launchpad (that one is the step sequencer)
            name = next((n for n in names if "launchpad" not in n.lower()), names[0])
        self.port = mido.open_input(name)
        self.port_name = name
        self.steering = steering
        self.map_path = map_path
        self.cc_map = {}  # (channel, cc) -> knob index
        if os.path.exists(map_path):
            with open(map_path) as f:
                self.cc_map = {tuple(map(int, k.split(":"))): v for k, v in json.load(f).items()}
        print(f"[midi] {name}: {len(self.cc_map)}/{N_KNOBS} knobs mapped")
        self._prompt_next()

    def _prompt_next(self):
        if len(self.cc_map) < N_KNOBS:
            print(f"[midi] learn: turn {AxisSteering.knob_name(len(self.cc_map))}")

    def _save(self):
        with open(self.map_path, "w") as f:
            json.dump({f"{c}:{cc}": i for (c, cc), i in self.cc_map.items()}, f, indent=1)

    def reset_learn(self):
        self.cc_map = {}
        if os.path.exists(self.map_path):
            os.remove(self.map_path)
        print("[midi] mapping cleared")
        self._prompt_next()

    def run(self):
        for msg in self.port:
            if msg.type != "control_change":
                continue
            key = (msg.channel, msg.control)
            if key not in self.cc_map and len(self.cc_map) < N_KNOBS:
                idx = len(self.cc_map)
                self.cc_map[key] = idx
                self._save()
                print(f"[midi] ch{msg.channel} CC{msg.control} -> {AxisSteering.knob_name(idx)}")
                self._prompt_next()
            if key in self.cc_map:
                self.steering.set_knob(self.cc_map[key], msg.value / 127)
