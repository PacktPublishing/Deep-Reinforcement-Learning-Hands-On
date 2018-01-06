from unittest import TestCase

import libbots.data
from libbots import data, subtitles


class TestData(TestCase):
    emb_dict = {
        data.BEGIN_TOKEN: 0,
        data.END_TOKEN: 1,
        data.UNKNOWN_TOKEN: 2,
        'a': 3,
        'b': 4
    }

    def test_encode_words(self):
        res = data.encode_words(['a', 'b', 'c'], self.emb_dict)
        self.assertEqual(res, [0, 3, 4, 2, 1])

    # def test_dialogues_to_train(self):
    #     dialogues = [
    #         [
    #             libbots.data.Phrase(words=['a', 'b'], time_start=0, time_stop=1),
    #             libbots.data.Phrase(words=['b', 'a'], time_start=2, time_stop=3),
    #             libbots.data.Phrase(words=['b', 'a'], time_start=2, time_stop=3),
    #         ],
    #         [
    #             libbots.data.Phrase(words=['a', 'b'], time_start=0, time_stop=1),
    #         ]
    #     ]
    #
    #     res = data.dialogues_to_train(dialogues, self.emb_dict)
    #     self.assertEqual(res, [
    #         ([0, 3, 4, 1], [0, 4, 3, 1]),
    #         ([0, 4, 3, 1], [0, 4, 3, 1]),
    #     ])
