"""Unit test for `data/utils.py`"""
from fractions import Fraction
from pathlib import Path
import unittest

from hydra import initialize, compose
from hydra.utils import to_absolute_path
from mido import MidiFile, MidiTrack, Message
import numpy as np

from data.utils import (Event, Converter, Modifier, process_tokens,
                        midi_to_events, events_to_midi, simplify)


class TestDataUtils(unittest.TestCase):
    """Tester for `data/utils.py`."""
    def setUp(self) -> None:
        with initialize(config_path="../config"):
            cfg = compose(config_name="main",
                          overrides=["model=music", "data=music"])
        self.cfg = cfg
        data_dir = Path(*cfg.data.data_dir)
        filename_list = Path(*cfg.data.filename_list)
        process_dir = Path(*cfg.data.process_dir)
        process_tokens(data_dir=data_dir,
                       filename_list=filename_list,
                       process_dir=process_dir)
        file_path = to_absolute_path(filename_list.joinpath("midi.txt"))
        with open(file_path, mode="r", encoding="utf-8") as file:
            self.path_list = file.readlines()
        self.num_special = cfg.data.num_special
        self.num_tokens = cfg.data.num_special + cfg.data.num_program \
                            + cfg.data.num_note + cfg.data.num_velocity \
                            + cfg.data.num_time_num + cfg.data.num_time_den

    def test_midi_to_events(self):
        """Tests `midi_to_events`.

        A sample MIDI file is constructed to test the function.
        This MIDI file consists of 1 track with 1 PROGRAM_CHANGE event,
        5 NOTE_ON events and 5 NOTE_OFF events.

        Track 1:
            - PROGRAM_CHANGE: channel=0, program=8, tick=0
            - NOTE_ON: channel=0, note=60, velocity=100, tick=0
            - NOTE_ON: channel=0, note=0, velocity=64, tick=120
            - NOTE_OFF: channel=0, note=0, velocity=0, tick=240
            - NOTE_ON: channel=0, note=30, velocity=48, tick=360
            - NOTE_ON: channel=0, note=72, velocity=72, tick=360
            - NOTE_OFF: channel=0, note=60, velocity=0, tick=480
            - NOTE_ON: channel=0, note=0, velocity=100, tick=600
            - NOTE_OFF: channel=0, note=0, velocity=0, tick=720
            - NOTE_OFF: channel=0, note=30, velocity=0, tick=720
            - NOTE_OFF: channel=0, note=72, velocity=0, tick=720

        ticks_per_beat is set to 480.

        The resulting events should be as follows:
        [
            Event(program=0, note=60, velocity=100, time=Fraction(0)),
            Event(program=0, note=0, velocity=64, time=Fraction(1 / 4)),
            Event(program=0, note=0, velocity=0, time=Fraction(1 / 2)),
            Event(program=0, note=30, velocity=48, time=Fraction(3 / 4)),
            Event(program=0, note=72, velocity=72, time=Fraction(3 / 4)),
            Event(program=0, note=60, velocity=0, time=Fraction(1)),
            Event(program=0, note=0, velocity=100, time=Fraction(5 / 4)),
            Event(program=0, note=0, velocity=0, time=Fraction(3 / 2)),
            Event(program=0, note=30, velocity=0, time=Fraction(3 / 2)),
            Event(program=0, note=72, velocity=0, time=Fraction(3 / 2)),
        ]"""

        # Construct a sample MIDI file.
        midi_file = MidiFile()
        midi_file.ticks_per_beat = 480
        midi_track = MidiTrack()
        midi_file.tracks.append(midi_track)
        midi_track.append(
            Message("program_change", channel=0, program=0, time=0))
        midi_track.append(
            Message("note_on", channel=0, note=60, velocity=100, time=0))
        midi_track.append(
            Message("note_on", channel=0, note=0, velocity=64, time=120))
        midi_track.append(
            Message("note_off", channel=0, note=0, velocity=0, time=120))
        midi_track.append(
            Message("note_on", channel=0, note=30, velocity=48, time=120))
        midi_track.append(
            Message("note_on", channel=0, note=72, velocity=72, time=0))
        midi_track.append(
            Message("note_off", channel=0, note=60, velocity=0, time=120))
        midi_track.append(
            Message("note_on", channel=0, note=0, velocity=100, time=120))
        midi_track.append(
            Message("note_off", channel=0, note=0, velocity=0, time=120))
        midi_track.append(
            Message("note_off", channel=0, note=30, velocity=0, time=0))
        midi_track.append(
            Message("note_off", channel=0, note=72, velocity=0, time=0))

        # Convert the MIDI file to events.
        events = midi_to_events(midi_file=midi_file)

        # Check the result.
        self.assertEqual(len(events), 10)
        # Check events.
        self.assertEqual(events, [
            Event(program=0, note=60, velocity=100, time=Fraction(0)),
            Event(program=0, note=0, velocity=64, time=Fraction(1 / 4)),
            Event(program=0, note=0, velocity=0, time=Fraction(1 / 2)),
            Event(program=0, note=30, velocity=48, time=Fraction(3 / 4)),
            Event(program=0, note=72, velocity=72, time=Fraction(3 / 4)),
            Event(program=0, note=60, velocity=0, time=Fraction(1)),
            Event(program=0, note=0, velocity=100, time=Fraction(5 / 4)),
            Event(program=0, note=0, velocity=0, time=Fraction(3 / 2)),
            Event(program=0, note=30, velocity=0, time=Fraction(3 / 2)),
            Event(program=0, note=72, velocity=0, time=Fraction(3 / 2))
        ])

    def test_events_to_midi(self):
        """Test `events_to_midi`.

        See test_midi_to_events(self) for test case."""

        # Initialize list of events.
        events = [
            Event(program=0, note=60, velocity=100, time=Fraction(0)),
            Event(program=0, note=0, velocity=64, time=Fraction(1 / 4)),
            Event(program=0, note=0, velocity=0, time=Fraction(1 / 2)),
            Event(program=0, note=30, velocity=48, time=Fraction(3 / 4)),
            Event(program=0, note=72, velocity=72, time=Fraction(3 / 4)),
            Event(program=0, note=60, velocity=0, time=Fraction(1)),
            Event(program=0, note=0, velocity=100, time=Fraction(5 / 4)),
            Event(program=0, note=0, velocity=0, time=Fraction(3 / 2)),
            Event(program=0, note=30, velocity=0, time=Fraction(3 / 2)),
            Event(program=0, note=72, velocity=0, time=Fraction(3 / 2)),
        ]
        # Convert the events to a MIDI file.
        midi_file = events_to_midi(events=events)
        # Check the result.
        self.assertEqual(midi_file.ticks_per_beat, 120)
        self.assertEqual(len(midi_file.tracks), 16)
        # Get track 10.
        track = midi_file.tracks[10]
        self.assertEqual(len(track), 11)
        self.assertEqual(track[0].type, "program_change")
        self.assertEqual(track[0].channel, 10)
        self.assertEqual(track[0].program, 0)
        self.assertEqual(track[0].time, 0)
        self.assertEqual(track[1].type, "note_on")
        self.assertEqual(track[1].channel, 10)
        self.assertEqual(track[1].note, 60)
        self.assertEqual(track[1].velocity, 100)
        self.assertEqual(track[1].time, 0)
        self.assertEqual(track[2].type, "note_on")
        self.assertEqual(track[2].channel, 10)
        self.assertEqual(track[2].note, 0)
        self.assertEqual(track[2].velocity, 64)
        self.assertEqual(track[2].time, 30)
        self.assertEqual(track[3].type, "note_off")
        self.assertEqual(track[3].channel, 10)
        self.assertEqual(track[3].note, 0)
        self.assertEqual(track[3].velocity, 0)
        self.assertEqual(track[3].time, 60)
        self.assertEqual(track[4].type, "note_on")
        self.assertEqual(track[4].channel, 10)
        self.assertEqual(track[4].note, 30)
        self.assertEqual(track[4].velocity, 48)
        self.assertEqual(track[4].time, 90)
        self.assertEqual(track[5].type, "note_on")
        self.assertEqual(track[5].channel, 10)
        self.assertEqual(track[5].note, 72)
        self.assertEqual(track[5].velocity, 72)
        self.assertEqual(track[5].time, 90)
        self.assertEqual(track[6].type, "note_off")
        self.assertEqual(track[6].channel, 10)
        self.assertEqual(track[6].note, 60)
        self.assertEqual(track[6].velocity, 0)
        self.assertEqual(track[6].time, 120)
        self.assertEqual(track[7].type, "note_on")
        self.assertEqual(track[7].channel, 10)
        self.assertEqual(track[7].note, 0)
        self.assertEqual(track[7].velocity, 100)
        self.assertEqual(track[7].time, 150)
        self.assertEqual(track[8].type, "note_off")
        self.assertEqual(track[8].channel, 10)
        self.assertEqual(track[8].note, 0)
        self.assertEqual(track[8].velocity, 0)
        self.assertEqual(track[8].time, 180)
        self.assertEqual(track[9].type, "note_off")
        self.assertEqual(track[9].channel, 10)
        self.assertEqual(track[9].note, 30)
        self.assertEqual(track[9].velocity, 0)
        self.assertEqual(track[9].time, 180)
        self.assertEqual(track[10].type, "note_off")
        self.assertEqual(track[10].channel, 10)
        self.assertEqual(track[10].note, 72)
        self.assertEqual(track[10].velocity, 0)
        self.assertEqual(track[10].time, 180)

    def test_simplify(self):
        """Tests `simplify`.

        `simplify(lower, upper)` returns the simplest fraction within the
        interval [lower, upper] where lower and upper are non-negative floats.

        The 'simplest` fraction is defined as the following:
            - If lower <= 0 <= upper, 0.
            - If 0 < lower <= 1 <= upper, 1.
            - If else, then Fraction(p, q) where among all
              lower <= p / q <= upper, q is the smallest.
              If multiple p's exist, the smallest p is selected."""

        # Test 1: lower <= 0 <= upper.
        self.assertEqual(simplify(lower=0., upper=0.), Fraction(0))
        self.assertEqual(simplify(lower=0., upper=0.5), Fraction(0))
        self.assertEqual(simplify(lower=0., upper=1.), Fraction(0))
        # Test 2: 0 < lower <= 1 <= upper.
        self.assertEqual(simplify(lower=0.5, upper=1.), Fraction(1))
        self.assertEqual(simplify(lower=0.5, upper=1.5), Fraction(1))
        self.assertEqual(simplify(lower=0.5, upper=2.), Fraction(1))
        # Test 3: else.
        self.assertEqual(simplify(lower=0.5, upper=0.75), Fraction(1, 2))
        self.assertEqual(simplify(lower=1 / 3, upper=2 / 3), Fraction(1, 2))
        self.assertEqual(simplify(lower=1 / 5, upper=2 / 7), Fraction(1, 4))
        self.assertEqual(simplify(lower=3 / 7, upper=4 / 9), Fraction(3, 7))
        self.assertEqual(simplify(lower=3 / 2, upper=10 / 3), Fraction(2))


class TestConverter(unittest.TestCase):
    """Tests for `Converter`."""
    def setUp(self):
        """Initializes `Converter`."""
        self.converter = Converter(token_type=np.uint8)

    def test_events_to_tokens(self):
        """Tests `Converter.events_to_tokens`.

        See `test_midi_to_events(self)` for event list.
        The resulting tokens should be as follows:

        [
            [0, 60, 100, 0, 1],
            [0, 0, 64, 1, 4],
            [0, 0, 0, 1, 4],
            [0, 30, 48, 1, 4],
            [0, 72, 72, 0, 1],
            [0, 60, 0, 1, 4],
            [0, 0, 100, 1, 4],
            [0, 0, 0, 1, 4],
            [0, 30, 0, 0, 1],
            [0, 72, 0, 0, 1],
        ]"""

        # Initialize list of events.
        events = [
            Event(program=0, note=60, velocity=100, time=Fraction(0)),
            Event(program=0, note=0, velocity=64, time=Fraction(1 / 4)),
            Event(program=0, note=0, velocity=0, time=Fraction(1 / 2)),
            Event(program=0, note=30, velocity=48, time=Fraction(3 / 4)),
            Event(program=0, note=72, velocity=72, time=Fraction(3 / 4)),
            Event(program=0, note=60, velocity=0, time=Fraction(1)),
            Event(program=0, note=0, velocity=100, time=Fraction(5 / 4)),
            Event(program=0, note=0, velocity=0, time=Fraction(3 / 2)),
            Event(program=0, note=30, velocity=0, time=Fraction(3 / 2)),
            Event(program=0, note=72, velocity=0, time=Fraction(3 / 2)),
        ]
        # Convert events to tokens.
        tokens = self.converter.events_to_tokens(events=events)
        print(tokens)
        # Check tokens.
        self.assertTrue(
            np.all(tokens == np.array([
                [0, 60, 100, 0, 1],
                [0, 0, 64, 1, 4],
                [0, 0, 0, 1, 4],
                [0, 30, 48, 1, 4],
                [0, 72, 72, 0, 1],
                [0, 60, 0, 1, 4],
                [0, 0, 100, 1, 4],
                [0, 0, 0, 1, 4],
                [0, 30, 0, 0, 1],
                [0, 72, 0, 0, 1],
            ])))

    def test_tokens_to_events(self):
        """Tests `Converter.tokens_to_events`.

        See `TestDataUtils` for token list.
        The resulting events should be as follows:

        [
            Event(program=0, note=60, velocity=100, time=Fraction(0)),
            Event(program=0, note=0, velocity=64, time=Fraction(1 / 4)),
            Event(program=0, note=0, velocity=0, time=Fraction(1 / 2)),
            Event(program=0, note=30, velocity=48, time=Fraction(3 / 4)),
            Event(program=0, note=72, velocity=72, time=Fraction(3 / 4)),
            Event(program=0, note=60, velocity=0, time=Fraction(1)),
            Event(program=0, note=0, velocity=100, time=Fraction(5 / 4)),
            Event(program=0, note=0, velocity=0, time=Fraction(3 / 2)),
            Event(program=0, note=30, velocity=0, time=Fraction(3 / 2)),
            Event(program=0, note=72, velocity=0, time=Fraction(3 / 2)),
        ]"""

        # Initialize list of tokens.
        tokens = np.array([
            [0, 60, 100, 0, 1],
            [0, 0, 64, 1, 4],
            [0, 0, 0, 1, 4],
            [0, 30, 48, 1, 4],
            [0, 72, 72, 0, 1],
            [0, 60, 0, 1, 4],
            [0, 0, 100, 1, 4],
            [0, 0, 0, 1, 4],
            [0, 30, 0, 0, 1],
            [0, 72, 0, 0, 1],
        ])
        # Convert tokens to events.
        events = self.converter.tokens_to_events(tokens=tokens)
        # Check events.
        self.assertEqual(events, [
            Event(program=0, note=60, velocity=100, time=Fraction(0)),
            Event(program=0, note=0, velocity=64, time=Fraction(1 / 4)),
            Event(program=0, note=0, velocity=0, time=Fraction(1 / 2)),
            Event(program=0, note=30, velocity=48, time=Fraction(3 / 4)),
            Event(program=0, note=72, velocity=72, time=Fraction(3 / 4)),
            Event(program=0, note=60, velocity=0, time=Fraction(1)),
            Event(program=0, note=0, velocity=100, time=Fraction(5 / 4)),
            Event(program=0, note=0, velocity=0, time=Fraction(3 / 2)),
            Event(program=0, note=30, velocity=0, time=Fraction(3 / 2)),
            Event(program=0, note=72, velocity=0, time=Fraction(3 / 2))
        ])


class TestModifier(unittest.TestCase):
    """Tests for `Modifier`."""
    def setUp(self):
        """Initializes `Modifier`."""
        with initialize(config_path="../config"):
            cfg = compose(config_name="main",
                          overrides=["model=music", "data=music"])
        self.cfg = cfg
        self.modifier = Modifier(num_special=cfg.data.num_special,
                                 num_program=cfg.data.num_program,
                                 num_note=cfg.data.num_note,
                                 num_velocity=cfg.data.num_velocity,
                                 num_time_num=cfg.data.num_time_num,
                                 note_shift=cfg.data.note_shift,
                                 velocity_scale=cfg.data.velocity_scale,
                                 time_scale=cfg.data.time_scale)

    def test_augment(self):
        """Tests `Modifier.augment`.

        See `TestDataUtils` for token list.
        The augmented tokens should have the same program, time_num,
        and time_den.
        The augmented notes and velocities should be within bounds."""

        # Initialize list of tokens.
        tokens = np.array([
            [0, 60, 100, 0, 1],
            [0, 0, 64, 1, 4],
            [0, 0, 0, 1, 4],
            [0, 30, 48, 1, 4],
            [0, 72, 72, 0, 1],
            [0, 60, 0, 1, 4],
            [0, 0, 100, 1, 4],
            [0, 0, 0, 1, 4],
            [0, 30, 0, 0, 1],
            [0, 72, 0, 0, 1],
        ])
        # Get augmented tokens.
        augmented_tokens = self.modifier.augment(tokens)
        # Programs should be the same.
        self.assertTrue(np.all(tokens[:, 0] == augmented_tokens[:, 0]))
        # time_num should be the same.
        self.assertTrue(np.all(tokens[:, 3] == augmented_tokens[:, 3]))
        # time_den should be the same.
        self.assertTrue(np.all(tokens[:, 4] == augmented_tokens[:, 4]))
        # Notes should be within bounds.
        self.assertTrue(np.all(0 <= tokens[:, 1]))
        self.assertTrue(np.all(tokens[:, 1] < self.cfg.data.num_note))
        # Velocities should be within bounds.
        self.assertTrue(np.all(0 <= tokens[:, 2]))
        self.assertTrue(np.all(tokens[:, 2] < self.cfg.data.num_velocity))

    def test_flatten(self):
        """Tests `Modifier.flatten`.

        See `TestDataUtils` for tokens list.
        The flattened tokens should be 1-dimensional.
        The calculated positions should have 3 columns."""

        # Initialize list of tokens.
        tokens = np.array([
            [0, 60, 100, 0, 1],
            [0, 0, 64, 1, 4],
            [0, 0, 0, 1, 4],
            [0, 30, 48, 1, 4],
            [0, 72, 72, 0, 1],
            [0, 60, 0, 1, 4],
            [0, 0, 100, 1, 4],
            [0, 0, 0, 1, 4],
            [0, 30, 0, 0, 1],
            [0, 72, 0, 0, 1],
        ])
        # Flatten tokens.
        flattened_tokens, positions = self.modifier.flatten(tokens)
        # Compare tokens.
        self.assertTrue(
            np.all(flattened_tokens == np.array([
                3, 192, 360, 389, 648, 3, 132, 324, 389, 648, 3, 132, 389, 648,
                3, 162, 308, 3, 204, 332, 389, 648, 3, 192, 389, 648, 3, 132,
                360, 389, 648, 3, 132, 3, 162, 3, 204
            ])))
        # Compare time positions.
        self.assertTrue(
            np.all(positions[:, 0] == np.array([
                0, 0, 0, 0, 0, 0.25, 0.25, 0.25, 0.25, 0.25, 0.5, 0.5, 0.5,
                0.5, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 1, 1, 1,
                1, 1.25, 1.25, 1.25, 1.25, 1.25, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5
            ]) * self.cfg.data.time_scale))
        # Compare note positions.
        self.assertTrue(
            np.all(positions[:, 1] == np.array([
                0, 0, 0, 60, 60, 60, 60, 60, 0, 0, 0, 0, 0, 0, 0, 0, 0, 30, 30,
                30, 72, 72, 72, 72, 60, 60, 60, 60, 60, 0, 0, 0, 0, 0, 0, 30,
                30
            ],
                                               dtype=np.float32)))
        # Compare velocity positions.
        self.assertTrue(
            np.all(positions[:, 2] == np.array([
                0, 0, 0, 100, 100, 100, 100, 100, 64, 64, 64, 64, 0, 0, 0, 0,
                0, 48, 48, 48, 72, 72, 72, 72, 0, 0, 0, 0, 0, 100, 100, 100,
                100, 0, 0, 0, 0
            ],
                                               dtype=np.float32)))

    def test_unflatten(self):
        """Tests `Modifier.unflatten`.

        See `test_flatten` for tokens list.
        The unflattened tokens should be 2-dimensional."""

        # Initialize list of tokens.
        tokens = np.array([
            3, 192, 360, 389, 648, 3, 132, 324, 389, 648, 3, 132, 389, 648, 3,
            162, 308, 3, 204, 332, 389, 648, 3, 192, 389, 648, 3, 132, 360,
            389, 648, 3, 132, 3, 162, 3, 204
        ])
        # Unflatten tokens.
        unflattened_tokens = self.modifier.unflatten(tokens)
        # Compare unflattened tokens.
        self.assertTrue(
            np.all(unflattened_tokens == np.array([
                [0, 60, 100, 0, 1],
                [0, 0, 64, 1, 4],
                [0, 0, 0, 1, 4],
                [0, 30, 48, 1, 4],
                [0, 72, 72, 0, 1],
                [0, 60, 0, 1, 4],
                [0, 0, 100, 1, 4],
                [0, 0, 0, 1, 4],
                [0, 30, 0, 0, 1],
                [0, 72, 0, 0, 1],
            ])))

    def test_pad_or_slice(self):
        """Tests `Modifier.pad_or_slice`.

        See `test_flatten` for tokens and positions.
        Resulting tokens and positions should have length `length`.
        Tokens should be bookended with BEGIN and END tokens.
        Positions should be padded with positions[0] and positions[-1]."""

        # Initialize list of tokens.
        tokens = np.array([
            3, 192, 360, 389, 648, 3, 132, 324, 389, 648, 3, 132, 389, 648, 3,
            162, 308, 3, 204, 332, 389, 648, 3, 192, 389, 648, 3, 132, 360,
            389, 648, 3, 132, 3, 162, 3, 204
        ])
        # Initialize positions.
        positions = np.array([
            [0., 0., 0.],
            [0., 0., 0.],
            [0., 0., 0.],
            [0., 60., 100.],
            [0., 60., 100.],
            [0.25, 60., 100.],
            [0.25, 60., 100.],
            [0.25, 60., 100.],
            [0.25, 0., 64.],
            [0.25, 0., 64.],
            [0.5, 0., 64.],
            [0.5, 0., 64.],
            [0.5, 0., 0.],
            [0.5, 0., 0.],
            [0.75, 0., 0.],
            [0.75, 0., 0.],
            [0.75, 0., 0.],
            [0.75, 30., 48.],
            [0.75, 30., 48.],
            [0.75, 30., 48.],
            [0.75, 72., 72.],
            [0.75, 72., 72.],
            [1., 72., 72.],
            [1., 72., 72.],
            [1., 60., 0.],
            [1., 60., 0.],
            [1.25, 60., 0.],
            [1.25, 60., 0.],
            [1.25, 60., 0.],
            [1.25, 0., 100.],
            [1.25, 0., 100.],
            [1.5, 0., 100.],
            [1.5, 0., 100.],
            [1.5, 0., 0.],
            [1.5, 0., 0.],
            [1.5, 30., 0.],
            [1.5, 30., 0.],
        ])
        # Pad or slice tokens.
        padded_tokens, padded_positions = self.modifier.pad_or_slice(
            tokens, positions, length=self.cfg.model.data_len)
        # Compare padded tokens.
        self.assertEqual(len(padded_tokens), self.cfg.model.data_len)
        self.assertEqual(padded_tokens[0], self.modifier.begin)
        self.assertEqual(padded_tokens[len(tokens) + 1], self.modifier.end)
        for i in range(len(tokens) + 2, self.cfg.model.data_len):
            self.assertEqual(padded_tokens[i], self.modifier.pad)
        # Compare padded positions.
        self.assertEqual(len(padded_positions), self.cfg.model.data_len)
        self.assertTrue(np.all(padded_positions[0] == positions[0]))
        for i in range(len(positions) + 1, self.cfg.model.data_len):
            self.assertTrue(np.all(padded_positions[i] == positions[-1]))
