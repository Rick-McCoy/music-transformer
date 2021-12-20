import os
import random
import unittest

from hydra import initialize, compose
from hydra.utils import to_absolute_path
from mido.midifiles.midifiles import MidiFile
import numpy as np

from data.utils import MessageType, Note, Tokenizer, read_midi, prepare_data, write_midi


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

    def test_read_midi(self):
        self.assertTrue(self.path_list)
        valid_types = [MessageType.NOTE_OFF, MessageType.NOTE_ON]
        for path in self.path_list[:10]:
            filename = to_absolute_path(
                os.path.join(*self.cfg.data.data_dir, path.strip()))
            notes = read_midi(MidiFile(filename=filename, clip=True))
            prev_tick = 0
            for note in notes:
                self.assertIn(note.type, valid_types)
                tick_delta = note.tick - prev_tick
                self.assertGreaterEqual(tick_delta, 0)
                prev_tick = note.tick
                self.assertGreaterEqual(note.pitch, 0)
                self.assertLess(note.pitch, self.cfg.model.num_pitch)
                self.assertGreaterEqual(note.velocity, 0)
                self.assertLess(note.velocity, self.cfg.model.num_velocity)
                self.assertGreaterEqual(note.program, 0)
                self.assertLess(note.program, self.cfg.model.num_program)

    def test_tokenize(self):
        self.assertTrue(self.path_list)
        for path in self.path_list[:10]:
            filename = to_absolute_path(
                os.path.join(*self.cfg.data.data_dir, path.strip()))
            data = self.tokenizer.tokenize(
                read_midi(MidiFile(filename=filename, clip=True)))
            self.assertTrue(np.all(data >= 0))
            self.assertTrue(np.all(data < self.cfg.model.num_token))

    def test_write_midi(self):
        note_list = []
        note_list.append(
            Note(message_type=MessageType.NOTE_ON,
                 tick=0,
                 pitch=64,
                 velocity=64,
                 program=0))
        note_list.append(
            Note(message_type=MessageType.NOTE_ON,
                 tick=2,
                 pitch=48,
                 velocity=64,
                 program=8))
        note_list.append(
            Note(message_type=MessageType.NOTE_OFF,
                 tick=10,
                 pitch=64,
                 velocity=0,
                 program=0))
        note_list.append(
            Note(message_type=MessageType.NOTE_OFF,
                 tick=12,
                 pitch=48,
                 velocity=0,
                 program=8))
        midi_file = write_midi(note_list)
        for track in midi_file.tracks:
            self.assertEqual(track[0].type, "program_change")
            self.assertEqual(track[0].channel, 0)
            self.assertEqual(track[0].program, 0)
            self.assertEqual(track[0].time, 0)
            self.assertEqual(track[1].type, "note_on")
            self.assertEqual(track[1].note, 64)
            self.assertEqual(track[1].channel, 0)
            self.assertEqual(track[1].velocity, 64)
            self.assertEqual(track[1].time, 0)
            self.assertEqual(track[2].type, "program_change")
            self.assertEqual(track[2].channel, 0)
            self.assertEqual(track[2].program, 8)
            self.assertEqual(track[2].time, 2)
            self.assertEqual(track[3].type, "note_on")
            self.assertEqual(track[3].note, 48)
            self.assertEqual(track[3].channel, 0)
            self.assertEqual(track[3].velocity, 64)
            self.assertEqual(track[3].time, 0)
            self.assertEqual(track[4].type, "program_change")
            self.assertEqual(track[4].channel, 0)
            self.assertEqual(track[4].program, 0)
            self.assertEqual(track[4].time, 8)
            self.assertEqual(track[5].type, "note_off")
            self.assertEqual(track[5].note, 64)
            self.assertEqual(track[5].channel, 0)
            self.assertEqual(track[5].velocity, 0)
            self.assertEqual(track[5].time, 0)
            self.assertEqual(track[6].type, "program_change")
            self.assertEqual(track[6].channel, 0)
            self.assertEqual(track[6].program, 8)
            self.assertEqual(track[6].time, 2)
            self.assertEqual(track[7].type, "note_off")
            self.assertEqual(track[7].note, 48)
            self.assertEqual(track[7].channel, 0)
            self.assertEqual(track[7].velocity, 0)
            self.assertEqual(track[7].time, 0)
