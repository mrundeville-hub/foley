"""Behavioral checks for material synthesis; perceptual quality needs listening."""
import copy
import unittest

import numpy as np

from foley_engine import FPS, SR, MASTER_PEAK, render
from scipy.signal import resample_poly


def fixture(palette="metal", moving=False):
    count = 90
    motion = {k: np.full(count, .5) for k in ("cx", "cy", "rx", "ry", "lum")}
    motion.update(ms=np.zeros(count), reg=np.zeros(count))
    if moving:
        motion["ms"][12:65] = .02 + .015 * np.sin(np.linspace(0, 5, 53))
        motion["reg"] = motion["ms"].copy()
    analysis = dict(T=count, motion=motion, scenes=[dict(a=0, b=count / FPS, palette=palette)])
    plan = dict(vibe="resonant", cues=[dict(type="hit", t=.5, s=.8, x=.3)])
    return analysis, plan


class MaterialEngineTests(unittest.TestCase):
    def run_sound(self, analysis, plan, seed=42):
        return render(copy.deepcopy(analysis), copy.deepcopy(plan), seed, ([], []))

    def test_seed_repeats_and_variation_changes_sound(self):
        an, plan = fixture(moving=True)
        a = self.run_sound(an, plan)
        b = self.run_sound(an, plan)
        c = self.run_sound(an, plan, 43)
        np.testing.assert_array_equal(a.audio, b.audio)
        self.assertGreater(np.sqrt(np.mean((a.audio - c.audio) ** 2)), .02)

    def test_musical_key_does_not_retune_material(self):
        an, plan = fixture()
        a = self.run_sound(an, dict(plan, key="C", mode="major"))
        b = self.run_sound(an, dict(plan, key="F#", mode="minor"))
        np.testing.assert_array_equal(a.audio, b.audio)

    def test_static_picture_is_silent_without_contact(self):
        an, plan = fixture()
        plan["cues"] = []
        result = self.run_sound(an, plan)
        self.assertEqual(np.count_nonzero(result.audio), 0)
        self.assertFalse(result.events)
        self.assertTrue(all(np.count_nonzero(m) == 0 for m in result.meters.values()))

    def test_contact_starts_at_cue_and_decays(self):
        an, plan = fixture()
        result = self.run_sound(an, plan)
        onset = int(.5 * SR)
        # Oversampling has sub-sample pre-ringing; it must stay below half a PCM16 step.
        self.assertLess(np.max(np.abs(result.audio[:, :onset])), .5 / 32767)
        self.assertGreater(np.max(np.abs(result.audio[:, onset:onset + int(.08 * SR)])), .05)
        self.assertLess(np.sqrt(np.mean(result.audio[:, -SR // 2:] ** 2)), .001)
        for name in ("HIT", "VOICE", "GRAIN"):
            self.assertGreater(np.max(np.abs(result.stems[name])), .005, name)

    def test_materials_have_distinct_spectra(self):
        spectra = []
        for palette in ("metal", "glass", "organic"):
            an, plan = fixture(palette)
            mono = self.run_sound(an, plan).audio.mean(axis=0)
            power = np.abs(np.fft.rfft(mono)) ** 2
            frequencies = np.fft.rfftfreq(len(mono), 1 / SR)
            bands = np.array([power[(frequencies >= a) & (frequencies < b)].sum()
                              for a, b in ((20, 150), (150, 700), (700, 3000), (3000, 16000))])
            spectra.append(bands / bands.sum())
        for a, b in ((0, 1), (0, 2), (1, 2)):
            self.assertGreater(np.linalg.norm(spectra[a] - spectra[b]), .1)

    def test_dropouts_mute_output_and_meter(self):
        an, plan = fixture(moving=True)
        plan["cues"].append(dict(type="dropout", t=.8, d=.5, s=1))
        result = self.run_sound(an, plan)
        self.assertLess(np.max(np.abs(result.audio[:, int(.95 * SR):int(1.2 * SR)])), 1e-5)
        self.assertTrue(all(np.max(m[29:35]) < 1e-5 for m in result.meters.values()))

    def test_master_and_meters_are_finite_and_bounded(self):
        an, plan = fixture(moving=True)
        plan["cues"] += [dict(type="boom", t=t, s=1) for t in np.arange(.7, 2.5, .1)]
        result = self.run_sound(an, plan)
        self.assertEqual(result.audio.shape, (2, 3 * SR))
        self.assertTrue(np.isfinite(result.audio).all())
        self.assertLessEqual(np.max(np.abs(resample_poly(result.audio, 4, 1, axis=-1))), MASTER_PEAK + 1e-6)
        self.assertLessEqual(np.sqrt(np.mean(result.audio.astype(float) ** 2)), 10 ** (-16.6 / 20) + 1e-6)
        for meter in result.meters.values():
            self.assertEqual(len(meter), an["T"])
            self.assertTrue(np.isfinite(meter).all())
            self.assertLessEqual(meter.max(), 1)

    def test_gesture_has_directed_contour_and_quiet_tail(self):
        an, plan = fixture("organic")
        plan["cues"] = [dict(type="sweep", t=.5, d=.7, s=.8, pitch="up")]
        up = self.run_sound(an, plan)
        plan["cues"][0]["pitch"] = "down"
        down = self.run_sound(an, plan)
        a, b = up.events[0]["pitch_curve"], down.events[0]["pitch_curve"]
        self.assertGreater(max(a) - min(a), 7)
        self.assertGreater(a[len(a) // 2] - b[len(b) // 2], 10)
        self.assertGreater(np.sqrt(np.mean((up.audio - down.audio) ** 2)), .02)
        self.assertLess(np.sqrt(np.mean(up.audio[:, -SR:] ** 2)), .0001)


if __name__ == "__main__":
    unittest.main()
