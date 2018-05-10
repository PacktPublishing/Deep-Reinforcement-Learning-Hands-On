import datetime
from unittest import TestCase

import libbots.data
from libbots import subtitles


class TestPhrases(TestCase):
    def test_split_phrase(self):
        phrase = libbots.data.Phrase(words=["a", "b", "c"], time_start=datetime.timedelta(seconds=0),
                                     time_stop=datetime.timedelta(seconds=10))
        res = subtitles.split_phrase(phrase)
        self.assertIsInstance(res, list)
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0], phrase)

        phrase = libbots.data.Phrase(words=["a", "b", "-", "c"], time_start=datetime.timedelta(seconds=0),
                                     time_stop=datetime.timedelta(seconds=10))
        res = subtitles.split_phrase(phrase)
        self.assertEqual(len(res), 2)
        self.assertEqual(res[0].words, ["a", "b"])
        self.assertEqual(res[1].words, ["c"])
        self.assertAlmostEqual(res[0].time_start.total_seconds(), 0)
        self.assertAlmostEqual(res[0].time_stop.total_seconds(), 5)
        self.assertAlmostEqual(res[1].time_start.total_seconds(), 5)
        self.assertAlmostEqual(res[1].time_stop.total_seconds(), 10)

        phrase = libbots.data.Phrase(words=['-', 'Wait', 'a', 'sec', '.', '-'], time_start=datetime.timedelta(0, 588, 204000),
                                     time_stop=datetime.timedelta(0, 590, 729000))
        res = subtitles.split_phrase(phrase)
        self.assertEqual(res[0].words, ["Wait", "a", "sec", "."])


class TestUtils(TestCase):
    def test_parse_time(self):
        self.assertEqual(subtitles.parse_time("00:00:33,074"),
                         datetime.timedelta(seconds=33, milliseconds=74))

    def test_remove_braced_words(self):
        self.assertEqual(subtitles.remove_braced_words(['a', 'b', 'c']),
                         ['a', 'b', 'c'])
        self.assertEqual(subtitles.remove_braced_words(['a', '[', 'b', ']', 'c']),
                         ['a', 'c'])
        self.assertEqual(subtitles.remove_braced_words(['a', '[', 'b', 'c']),
                         ['a'])
        self.assertEqual(subtitles.remove_braced_words(['a', ']', 'b', 'c']),
                         ['a', 'b', 'c'])
        self.assertEqual(subtitles.remove_braced_words(['a', '(', 'b', ']', 'c']),
                         ['a', 'c'])
        self.assertEqual(subtitles.remove_braced_words(['a', '(', 'b', 'c']),
                         ['a'])
        self.assertEqual(subtitles.remove_braced_words(['a', ')', 'b', 'c']),
                         ['a', 'b', 'c'])
