import os
import random
import unittest

from hydra import initialize, compose
from hydra.utils import to_absolute_path
from mido.midifiles.midifiles import MidiFile
import numpy as np

from data.utils import Tokenizer, read_midi, prepare_data


class TestDataUtils(unittest.TestCase):
    def setUp(self) -> None:
        with initialize(config_path="../config"):
            cfg = compose(config_name="config")
        self.cfg = cfg
        data_dir = os.path.join(*self.cfg.data.data_dir)
        file_dir = os.path.join(*self.cfg.data.file_dir)
        prepare_data(data_dir, file_dir)
        file_path = to_absolute_path(os.path.join(file_dir, "midi.txt"))
        with open(file_path, mode="r", encoding="utf-8") as file:
            self.path_list = file.readlines()
        random.shuffle(self.path_list)
        self.tokenizer = Tokenizer(self.cfg)

    def test_tokenize(self):
        self.assertTrue(self.path_list)
        for path in self.path_list[:10]:
            filename = to_absolute_path(
                os.path.join(*self.cfg.data.data_dir, path.strip()))
            data = self.tokenizer.tokenize(
                read_midi(MidiFile(filename=filename, clip=True)))
            self.assertTrue(np.all(data >= 0))
            self.assertTrue(np.all(data < self.cfg.model.num_token))
