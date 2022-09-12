import unittest

import torch

from config.config import NUM_TOKEN
from model.model import MusicModel


class TestMusicModel(unittest.TestCase):
    def test_music_model(self):
        model = MusicModel(
            learning_rate=0.001,
            weight_decay=0.0001,
            d_model=256,
            data_len=512,
            dropout=0.1,
            feed_forward=512,
            nhead=8,
            num_layers=6,
        )
        random_input = torch.randint(0, NUM_TOKEN, (2, 512))
        random_output = model(random_input)
        self.assertEqual(random_output.shape, (2, NUM_TOKEN, 512))
