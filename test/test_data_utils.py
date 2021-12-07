import os
import random
import unittest

from hydra.compose import compose
from hydra.initialize import initialize
from hydra.utils import to_absolute_path
from mido.midifiles.midifiles import MidiFile
import numpy as np

from data.utils import read_midi, prepare_data


class TestDataUtils(unittest.TestCase):
    def setUp(self) -> None:
        with initialize(config_path="../config"):
            self.cfg = compose(config_name="config")
        prepare_data(self.cfg.data.data_dir, self.cfg.data.tar_dir)
        file_path = to_absolute_path(
            os.path.join(self.cfg.data.tar_dir, "midi.txt"))
        with open(file_path, mode="r", encoding="utf-8") as file:
            self.path_list = file.readlines()
        random.shuffle(self.path_list)

    def test_read_midi(self):
        self.assertTrue(self.path_list)
        for path in self.path_list[:10]:
            filename = to_absolute_path(
                os.path.join(self.cfg.data.data_dir, path.strip()))
            ticks, programs, types, pitches, velocities = read_midi(
                MidiFile(filename=filename, clip=True))
            self.assertEqual(ticks.shape, programs.shape)
            self.assertEqual(ticks.shape, types.shape)
            self.assertEqual(ticks.shape, pitches.shape)
            self.assertEqual(ticks.shape, velocities.shape)
            self.assertTrue(np.all(ticks >= 0))
            self.assertTrue(np.all(programs >= 0))
            self.assertTrue(np.all(programs < 128))
            self.assertTrue(np.all(types >= 1))
            self.assertTrue(np.all(types <= 2))
            self.assertTrue(np.all(pitches >= 0))
            self.assertTrue(np.all(pitches < 128))
            self.assertTrue(np.all(velocities >= 0))
            self.assertTrue(np.all(velocities < 128))
