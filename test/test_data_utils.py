from pathlib import Path
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
        file_dir = Path(*self.cfg.data.file_dir)
        prepare_data(cfg)
        file_path = to_absolute_path(file_dir.joinpath("midi.txt"))
        with open(file_path, mode="r", encoding="utf-8") as file:
            self.path_list = file.readlines()
        random.shuffle(self.path_list)
        self.tokenizer = Tokenizer(self.cfg)

    def test_read_midi(self):
        self.assertTrue(self.path_list)
        valid_types = [MessageType.NOTE_OFF, MessageType.NOTE_ON]
        for path in self.path_list[:10]:
            filename = to_absolute_path(
                Path(*self.cfg.data.data_dir, path.strip()))
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
        tokens = self.tokenizer.tokenize(note_list)
        self.assertEqual(tokens[0], 1)
        self.assertEqual(tokens[1], 132)
        self.assertEqual(tokens[2], 325)
        self.assertEqual(tokens[3], 453)
        self.assertEqual(tokens[4], 140)
        self.assertEqual(tokens[5], 309)
        self.assertEqual(tokens[6], 453)
        self.assertEqual(tokens[7], 518)
        self.assertEqual(tokens[8], 3)
        self.assertEqual(tokens[9], 325)
        self.assertEqual(tokens[10], 524)
        self.assertEqual(tokens[11], 11)
        self.assertEqual(tokens[12], 309)
        self.assertEqual(tokens[13], 518)
        self.assertEqual(tokens[14], 2)

    def test_tokens_to_notes(self):
        tokens = np.array([
            1, 132, 325, 453, 140, 309, 453, 518, 3, 325, 524, 11, 309, 518, 2
        ],
                          dtype=np.int64)
        note_list = self.tokenizer.tokens_to_notes(tokens)
        self.assertEqual(note_list[0].tick, 0)
        self.assertEqual(note_list[0].pitch, 64)
        self.assertEqual(note_list[0].velocity, 64)
        self.assertEqual(note_list[0].program, 0)
        self.assertEqual(note_list[1].tick, 2)
        self.assertEqual(note_list[1].pitch, 48)
        self.assertEqual(note_list[1].velocity, 64)
        self.assertEqual(note_list[1].program, 8)
        self.assertEqual(note_list[2].tick, 10)
        self.assertEqual(note_list[2].pitch, 64)
        self.assertEqual(note_list[2].velocity, 0)
        self.assertEqual(note_list[2].program, 0)
        self.assertEqual(note_list[3].tick, 12)
        self.assertEqual(note_list[3].pitch, 48)
        self.assertEqual(note_list[3].velocity, 0)
        self.assertEqual(note_list[3].program, 8)

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
            self.assertEqual(track[1].type, "program_change")
            self.assertEqual(track[1].channel, 1)
            self.assertEqual(track[1].program, 8)
            self.assertEqual(track[1].time, 0)
            self.assertEqual(track[2].type, "note_on")
            self.assertEqual(track[2].note, 64)
            self.assertEqual(track[2].channel, 0)
            self.assertEqual(track[2].velocity, 64)
            self.assertEqual(track[2].time, 0)
            self.assertEqual(track[3].type, "note_on")
            self.assertEqual(track[3].note, 48)
            self.assertEqual(track[3].channel, 1)
            self.assertEqual(track[3].velocity, 64)
            self.assertEqual(track[3].time, 2)
            self.assertEqual(track[4].type, "note_off")
            self.assertEqual(track[4].note, 64)
            self.assertEqual(track[4].channel, 0)
            self.assertEqual(track[4].velocity, 0)
            self.assertEqual(track[4].time, 8)
            self.assertEqual(track[5].type, "note_off")
            self.assertEqual(track[5].note, 48)
            self.assertEqual(track[5].channel, 1)
            self.assertEqual(track[5].velocity, 0)
            self.assertEqual(track[5].time, 2)
