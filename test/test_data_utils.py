import random
import unittest

import numpy as np
from hydra import compose, initialize
from mido.midifiles.midifiles import MidiFile

from config.config import NUM_DRUM, NUM_NOTE, NUM_PROGRAM, CustomConfig
from data.utils import Event, MessageType, Tokenizer, read_midi, write_midi


class TestDataUtils(unittest.TestCase):
    def setUp(self) -> None:
        with initialize(config_path="../config", version_base=None):
            cfg = compose(config_name="config")
        self.cfg = CustomConfig(cfg)
        file_path = self.cfg.file_dir / "midi.txt"
        with open(file_path, mode="r", encoding="utf-8") as file:
            self.path_list = file.readlines()
        random.shuffle(self.path_list)
        self.tokenizer = Tokenizer()

    def test_read_midi(self):
        self.assertTrue(self.path_list)
        for path in self.path_list[:25]:
            filename = self.cfg.data_dir / path.strip()
            midi_file = MidiFile(filename=filename, clip=True)
            events = read_midi(midi_file)
            prev_tick = 0
            for event in events:
                tick_delta = event.tick - prev_tick
                self.assertGreaterEqual(tick_delta, 0)
                prev_tick = event.tick
                if event.program is None:
                    self.assertIsNone(event.note)
                    assert event.drum is not None
                    self.assertLess(event.drum, NUM_DRUM)
                    self.assertEqual(event.type, MessageType.NOTE_ON)
                else:
                    self.assertLess(event.program, NUM_PROGRAM)
                    assert event.note is not None
                    self.assertLess(event.note, NUM_NOTE)
                    self.assertIsNone(event.drum)

    def test_tokenize(self):
        event_list = []
        event_list.append(
            Event(
                message_type=MessageType.NOTE_ON,
                tick=0,
                note=64,
                program=0,
            )
        )
        event_list.append(
            Event(
                message_type=MessageType.NOTE_ON,
                tick=2,
                note=48,
                program=8,
            )
        )
        event_list.append(
            Event(
                message_type=MessageType.NOTE_OFF,
                tick=10,
                note=64,
                program=0,
            )
        )
        event_list.append(
            Event(
                message_type=MessageType.NOTE_OFF,
                tick=12,
                note=48,
                program=8,
            )
        )
        tokens = self.tokenizer.tokenize(event_list)
        self.assertEqual(tokens[0], self.tokenizer.begin)
        self.assertEqual(tokens[1], self.tokenizer.note_on)
        self.assertEqual(tokens[2], self.tokenizer.special_limit + 0)
        self.assertEqual(tokens[3], self.tokenizer.drum_limit + 64)
        self.assertEqual(tokens[4], self.tokenizer.note_limit + 1)
        self.assertEqual(tokens[5], self.tokenizer.special_limit + 8)
        self.assertEqual(tokens[6], self.tokenizer.drum_limit + 48)
        self.assertEqual(tokens[7], self.tokenizer.note_limit + 7)
        self.assertEqual(tokens[8], self.tokenizer.note_off)
        self.assertEqual(tokens[9], self.tokenizer.special_limit + 0)
        self.assertEqual(tokens[10], self.tokenizer.drum_limit + 64)
        self.assertEqual(tokens[11], self.tokenizer.note_limit + 1)
        self.assertEqual(tokens[12], self.tokenizer.special_limit + 8)
        self.assertEqual(tokens[13], self.tokenizer.drum_limit + 48)
        self.assertEqual(tokens[14], self.tokenizer.end)

    def test_tokens_to_notes(self):
        tokens = np.array(
            [
                self.tokenizer.begin,
                self.tokenizer.note_on,
                self.tokenizer.special_limit + 0,
                self.tokenizer.drum_limit + 64,
                self.tokenizer.note_limit + 1,
                self.tokenizer.special_limit + 8,
                self.tokenizer.drum_limit + 48,
                self.tokenizer.note_limit + 7,
                self.tokenizer.note_off,
                self.tokenizer.special_limit + 0,
                self.tokenizer.drum_limit + 64,
                self.tokenizer.note_limit + 1,
                self.tokenizer.special_limit + 8,
                self.tokenizer.drum_limit + 48,
                self.tokenizer.end,
            ],
            dtype=np.int64,
        )
        event_list = self.tokenizer.tokens_to_events(tokens)
        self.assertEqual(event_list[0].type, MessageType.NOTE_ON)
        self.assertEqual(event_list[0].tick, 0)
        self.assertEqual(event_list[0].note, 64)
        self.assertEqual(event_list[0].program, 0)
        self.assertEqual(event_list[1].type, MessageType.NOTE_ON)
        self.assertEqual(event_list[1].tick, 2)
        self.assertEqual(event_list[1].note, 48)
        self.assertEqual(event_list[1].program, 8)
        self.assertEqual(event_list[2].type, MessageType.NOTE_OFF)
        self.assertEqual(event_list[2].tick, 10)
        self.assertEqual(event_list[2].note, 64)
        self.assertEqual(event_list[2].program, 0)
        self.assertEqual(event_list[3].type, MessageType.NOTE_OFF)
        self.assertEqual(event_list[3].tick, 12)
        self.assertEqual(event_list[3].note, 48)
        self.assertEqual(event_list[3].program, 8)

    def test_write_midi(self):
        event_list = []
        event_list.append(
            Event(
                message_type=MessageType.NOTE_ON,
                tick=0,
                note=64,
                program=0,
            )
        )
        event_list.append(
            Event(
                message_type=MessageType.NOTE_ON,
                tick=2,
                note=48,
                program=8,
            )
        )
        event_list.append(
            Event(
                message_type=MessageType.NOTE_OFF,
                tick=10,
                note=64,
                program=0,
            )
        )
        event_list.append(
            Event(
                message_type=MessageType.NOTE_OFF,
                tick=12,
                note=48,
                program=8,
            )
        )
        midi_file = write_midi(event_list)
        track = midi_file.tracks[0]
        self.assertEqual(len(track), 6)
        self.assertEqual(track[0].type, "program_change")
        self.assertEqual(track[0].channel, 10)
        self.assertEqual(track[0].program, 0)
        self.assertEqual(track[0].time, 0)
        self.assertEqual(track[1].type, "program_change")
        self.assertEqual(track[1].channel, 11)
        self.assertEqual(track[1].program, 8)
        self.assertEqual(track[1].time, 0)
        self.assertEqual(track[2].type, "note_on")
        self.assertEqual(track[2].note, 64)
        self.assertEqual(track[2].channel, 10)
        self.assertEqual(track[2].velocity, 64)
        self.assertEqual(track[2].time, 0)
        self.assertEqual(track[3].type, "note_on")
        self.assertEqual(track[3].note, 48)
        self.assertEqual(track[3].channel, 11)
        self.assertEqual(track[3].velocity, 64)
        self.assertEqual(track[3].time, round(2 / 120 * midi_file.ticks_per_beat))
        self.assertEqual(track[4].type, "note_off")
        self.assertEqual(track[4].note, 64)
        self.assertEqual(track[4].channel, 10)
        self.assertEqual(track[4].velocity, 0)
        self.assertEqual(track[4].time, round(8 / 120 * midi_file.ticks_per_beat))
        self.assertEqual(track[5].type, "note_off")
        self.assertEqual(track[5].note, 48)
        self.assertEqual(track[5].channel, 11)
        self.assertEqual(track[5].velocity, 0)
        self.assertEqual(track[5].time, round(2 / 120 * midi_file.ticks_per_beat))
