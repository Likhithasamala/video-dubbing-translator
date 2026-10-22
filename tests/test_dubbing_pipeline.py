import tempfile
import unittest
from pathlib import Path

import numpy as np

from dubbing import asr, dub, pipeline
from dubbing.languages import LANGUAGES, TARGET_CHOICES, get_language


class FakeEngine:
    """Pretends to be a TTS engine that always speaks for a fixed duration."""

    name = "fake"
    sample_rate = dub.SAMPLE_RATE

    def __init__(self, seconds):
        self.seconds = seconds

    def synthesize(self, text, speed=1.0, **_):
        return np.full(int(self.seconds * self.sample_rate), 0.3, dtype=np.float32), self.sample_rate


class LanguageTests(unittest.TestCase):
    def test_lookup_by_code_and_name(self):
        self.assertEqual(get_language("te").name, "Telugu")
        self.assertEqual(get_language("english").code, "en")
        self.assertIsNone(get_language("auto", allow_auto=True))
        with self.assertRaises(ValueError):
            get_language("klingon")

    def test_every_target_has_a_voice_and_translation_code(self):
        for language in LANGUAGES:
            self.assertRegex(language.nllb, r"^[a-z]{3}_[A-Z][a-z]{3}$")
        self.assertIn("Telugu", TARGET_CHOICES)
        self.assertIn("English", TARGET_CHOICES)
        self.assertNotIn("Japanese", TARGET_CHOICES)  # no free offline voice yet


class SegmentationTests(unittest.TestCase):
    def test_splits_on_sentence_end_and_long_pause(self):
        words = [(0.0, 0.5, " Hello"), (0.5, 1.2, " world."), (1.3, 1.8, " Next"),
                 (1.8, 2.6, " line"), (4.0, 4.5, " after"), (4.5, 5.6, " pause")]
        segments = asr._group_words(words)
        self.assertEqual([s.text for s in segments], ["Hello world.", "Next line", "after pause"])
        self.assertAlmostEqual(segments[2].start, 4.0)

    def test_tiny_fragment_is_merged_into_neighbour(self):
        words = [(0.0, 2.0, " A long first sentence."), (2.1, 2.4, " Yes.")]
        segments = asr._group_words(words)
        self.assertEqual(len(segments), 1)
        self.assertEqual(segments[0].end, 2.4)


class TimingTests(unittest.TestCase):
    def render(self, tts_seconds, segments, total):
        with tempfile.TemporaryDirectory() as work:
            return dub.render_voice_track(segments, FakeEngine(tts_seconds), total, work, log=lambda *_: None)

    def test_speech_starts_at_the_original_time(self):
        segment = asr.Segment(2.0, 4.0, "src", translation="dst")
        track = self.render(2.0, [segment], 6.0)
        self.assertEqual(track.size, 6 * dub.SAMPLE_RATE)
        voiced = np.flatnonzero(np.abs(track) > 0.01) / dub.SAMPLE_RATE
        self.assertAlmostEqual(voiced[0], 2.0, delta=0.05)
        self.assertAlmostEqual(voiced[-1], 4.0, delta=0.05)

    def test_long_line_is_compressed_and_never_overlaps_the_next(self):
        first = asr.Segment(0.0, 2.0, "a", translation="a")
        second = asr.Segment(3.0, 5.0, "b", translation="b")
        track = self.render(4.0, [first, second], 6.0)
        voiced = np.abs(track) > 0.01
        # First line (4 s of speech) must end before the second starts at 3.0 s.
        gap = voiced[int(2.96 * dub.SAMPLE_RATE):int(3.0 * dub.SAMPLE_RATE)]
        self.assertFalse(gap.all())
        self.assertTrue(voiced[int(3.1 * dub.SAMPLE_RATE)])


class LipSyncShotTests(unittest.TestCase):
    def tracks(self, boxes):
        from dubbing.lipsync import TRACK_COLUMNS

        data = np.full((len(boxes), TRACK_COLUMNS), np.nan, dtype=np.float32)
        for index, box in enumerate(boxes):
            if box is not None:
                data[index, :4] = box
        return data

    def test_face_runs_split_on_missing_face_and_hard_cut(self):
        from dubbing.lipsync import _face_runs

        face_a, face_b = (100, 100, 200, 200), (400, 100, 500, 200)
        tracks = self.tracks([face_a] * 5 + [None] * 2 + [face_a] * 3 + [face_b] * 4)
        self.assertEqual(_face_runs(tracks, 0, 14), [(0, 5), (7, 10), (10, 14)])

    def test_dub_tail_is_followed_only_until_the_next_line(self):
        from dubbing.lipsync import speech_frames

        fps, count = 10, 40
        speaking = np.zeros(count, dtype=bool)
        speaking[0:10] = True
        tracks = self.tracks([(0, 0, 50, 50)] * count)
        voice = np.ones(count * 100, dtype=np.float32)  # dub is talking the whole time
        segments = [asr.Segment(0.0, 1.0, "a"), asr.Segment(2.5, 3.5, "b")]
        active = speech_frames(speaking, tracks, segments, voice, 1000, fps)
        self.assertTrue(active[:25].all())
        self.assertFalse(active[25:].any())


class SubtitleTests(unittest.TestCase):
    def test_srt_format(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "out.srt"
            pipeline.write_srt([asr.Segment(1.5, 3.25, "x", translation="Hello")], path)
            self.assertEqual(path.read_text(encoding="utf-8"), "1\n00:00:01,500 --> 00:00:03,250\nHello\n")


if __name__ == "__main__":
    unittest.main()
