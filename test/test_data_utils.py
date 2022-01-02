from pathlib import Path
import random
import unittest

from hydra import initialize, compose
from hydra.utils import to_absolute_path
from mido.midifiles.midifiles import MidiFile
import numpy as np

from data.utils import MessageType, Event, Tokenizer, read_midi, prepare_data, write_midi


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
        valid_types = [
            MessageType.NOTE_OFF, MessageType.NOTE_ON,
            MessageType.CONTROL_CHANGE, MessageType.PITCHWHEEL
        ]
        for path in self.path_list[:10]:
            filename = to_absolute_path(
                Path(*self.cfg.data.data_dir, path.strip()))
            events = read_midi(MidiFile(filename=filename, clip=True))
            prev_tick = 0
            for event in events:
                self.assertIn(event.type, valid_types)
                tick_delta = event.tick - prev_tick
                self.assertGreaterEqual(tick_delta, 0)
                prev_tick = event.tick
                self.assertGreaterEqual(event.program, 0)
                self.assertLess(event.program, self.cfg.model.num_program)
                if event.type == MessageType.NOTE_ON:
                    self.assertGreaterEqual(event.note, 0)
                    self.assertLess(event.note, self.cfg.model.num_note)
                    self.assertGreaterEqual(event.velocity, 0)
                    self.assertLess(event.velocity,
                                    self.cfg.model.num_velocity)
                    self.assertIsNone(event.control)
                    self.assertIsNone(event.value)
                    self.assertIsNone(event.pitch)
                elif event.type == MessageType.NOTE_OFF:
                    self.assertGreaterEqual(event.note, 0)
                    self.assertLess(event.note, self.cfg.model.num_note)
                    self.assertIsNone(event.velocity)
                    self.assertIsNone(event.control)
                    self.assertIsNone(event.value)
                    self.assertIsNone(event.pitch)
                elif event.type == MessageType.CONTROL_CHANGE:
                    self.assertIsNone(event.note)
                    self.assertIsNone(event.velocity)
                    self.assertGreaterEqual(event.control, 0)
                    self.assertLess(event.control, self.cfg.model.num_control)
                    self.assertGreaterEqual(event.value, 0)
                    self.assertLess(event.value, self.cfg.model.num_value)
                    self.assertIsNone(event.pitch)
                elif event.type == MessageType.PITCHWHEEL:
                    self.assertIsNone(event.note)
                    self.assertIsNone(event.velocity)
                    self.assertIsNone(event.control)
                    self.assertIsNone(event.value)
                    self.assertGreaterEqual(event.pitch, -8192)
                    self.assertLess(event.pitch, 8192)

    def test_tokenize(self):
        self.assertEqual(
            self.cfg.model.num_special + self.cfg.model.num_program +
            self.cfg.model.num_note + self.cfg.model.num_velocity +
            self.cfg.model.num_control + self.cfg.model.num_value +
            self.cfg.model.num_pitch_1 + self.cfg.model.num_pitch_2 +
            self.cfg.model.num_tick, self.cfg.model.num_token)
        event_list = []
        event_list.append(
            Event(message_type=MessageType.NOTE_ON,
                  tick=0,
                  note=64,
                  velocity=64,
                  program=0))
        event_list.append(
            Event(message_type=MessageType.NOTE_ON,
                  tick=2,
                  note=48,
                  velocity=64,
                  program=8))
        event_list.append(
            Event(message_type=MessageType.NOTE_OFF,
                  tick=10,
                  note=64,
                  program=0))
        event_list.append(
            Event(message_type=MessageType.NOTE_OFF,
                  tick=12,
                  note=48,
                  program=8))
        event_list.append(
            Event(message_type=MessageType.CONTROL_CHANGE,
                  tick=14,
                  control=64,
                  value=64,
                  program=0))
        event_list.append(
            Event(message_type=MessageType.PITCHWHEEL,
                  tick=14,
                  pitch=-1022,
                  program=0))
        tokens = self.tokenizer.tokenize(event_list)
        self.assertEqual(tokens[0], 1)
        self.assertEqual(tokens[1], 3)
        self.assertEqual(tokens[2], 196)
        self.assertEqual(tokens[3], 452)
        self.assertEqual(tokens[4], 11)
        self.assertEqual(tokens[5], 180)
        self.assertEqual(tokens[6], 452)
        self.assertEqual(tokens[7], 1029)
        self.assertEqual(tokens[8], 3)
        self.assertEqual(tokens[9], 324)
        self.assertEqual(tokens[10], 1035)
        self.assertEqual(tokens[11], 11)
        self.assertEqual(tokens[12], 308)
        self.assertEqual(tokens[13], 1029)
        self.assertEqual(tokens[14], 3)
        self.assertEqual(tokens[15], 580)
        self.assertEqual(tokens[16], 708)
        self.assertEqual(tokens[17], 1029)
        self.assertEqual(tokens[18], 3)
        self.assertEqual(tokens[19], 828)
        self.assertEqual(tokens[20], 902)
        self.assertEqual(tokens[21], 2)

    def test_tokens_to_notes(self):
        tokens = np.array([
            1, 3, 196, 452, 11, 180, 452, 1029, 3, 324, 1035, 11, 308, 1029, 3,
            580, 708, 1029, 3, 828, 902, 2
        ],
                          dtype=np.int64)
        event_list = self.tokenizer.tokens_to_events(tokens)
        self.assertEqual(event_list[0].type, MessageType.NOTE_ON)
        self.assertEqual(event_list[0].tick, 0)
        self.assertEqual(event_list[0].note, 64)
        self.assertEqual(event_list[0].velocity, 64)
        self.assertIsNone(event_list[0].control)
        self.assertIsNone(event_list[0].value)
        self.assertIsNone(event_list[0].pitch)
        self.assertEqual(event_list[0].program, 0)
        self.assertEqual(event_list[1].type, MessageType.NOTE_ON)
        self.assertEqual(event_list[1].tick, 2)
        self.assertEqual(event_list[1].note, 48)
        self.assertEqual(event_list[1].velocity, 64)
        self.assertIsNone(event_list[1].control)
        self.assertIsNone(event_list[1].value)
        self.assertIsNone(event_list[1].pitch)
        self.assertEqual(event_list[1].program, 8)
        self.assertEqual(event_list[2].type, MessageType.NOTE_OFF)
        self.assertEqual(event_list[2].tick, 10)
        self.assertEqual(event_list[2].note, 64)
        self.assertIsNone(event_list[2].velocity)
        self.assertIsNone(event_list[2].control)
        self.assertIsNone(event_list[2].value)
        self.assertIsNone(event_list[2].pitch)
        self.assertEqual(event_list[2].program, 0)
        self.assertEqual(event_list[3].type, MessageType.NOTE_OFF)
        self.assertEqual(event_list[3].tick, 12)
        self.assertEqual(event_list[3].note, 48)
        self.assertIsNone(event_list[3].velocity)
        self.assertIsNone(event_list[3].control)
        self.assertIsNone(event_list[3].value)
        self.assertIsNone(event_list[3].pitch)
        self.assertEqual(event_list[3].program, 8)
        self.assertEqual(event_list[4].type, MessageType.CONTROL_CHANGE)
        self.assertEqual(event_list[4].tick, 14)
        self.assertIsNone(event_list[4].note)
        self.assertIsNone(event_list[4].velocity)
        self.assertEqual(event_list[4].control, 64)
        self.assertEqual(event_list[4].value, 64)
        self.assertIsNone(event_list[4].pitch)
        self.assertEqual(event_list[4].program, 0)
        self.assertEqual(event_list[5].type, MessageType.PITCHWHEEL)
        self.assertEqual(event_list[5].tick, 14)
        self.assertIsNone(event_list[5].note)
        self.assertIsNone(event_list[5].velocity)
        self.assertIsNone(event_list[5].control)
        self.assertIsNone(event_list[5].value)
        self.assertEqual(event_list[5].pitch, -1022)
        self.assertEqual(event_list[5].program, 0)

    def test_write_midi(self):
        event_list = []
        event_list.append(
            Event(message_type=MessageType.NOTE_ON,
                  tick=0,
                  note=64,
                  velocity=64,
                  program=0))
        event_list.append(
            Event(message_type=MessageType.NOTE_ON,
                  tick=2,
                  note=48,
                  velocity=64,
                  program=8))
        event_list.append(
            Event(message_type=MessageType.NOTE_OFF,
                  tick=10,
                  note=64,
                  velocity=0,
                  program=0))
        event_list.append(
            Event(message_type=MessageType.NOTE_OFF,
                  tick=12,
                  note=48,
                  velocity=0,
                  program=8))
        midi_file = write_midi(event_list)
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
