"""Implement various utility functions."""
from fractions import Fraction
import os
from pathlib import Path
from typing import List

from mido import MidiFile, MidiTrack
from mido.messages.messages import Message
from mido.midifiles.meta import KeySignatureError
import numpy as np
from numpy import ndarray
from tqdm import tqdm


class InvalidProgramError(Exception):
    """Exception raised for invalid program number."""
    def __init__(self, message):
        super().__init__()
        self.message = message


class InvalidNoteError(Exception):
    """Exception raised for invalid note number."""
    def __init__(self, message):
        super().__init__()
        self.message = message


class InvalidVelocityError(Exception):
    """Exception raised for invalid velocity."""
    def __init__(self, message):
        super().__init__()
        self.message = message


class InvalidPitchError(Exception):
    """Exception raised for invalid pitch."""
    def __init__(self, message):
        super().__init__()
        self.message = message


class InvalidTickError(Exception):
    """Exception raised for invalid tick."""
    def __init__(self, message):
        super().__init__()
        self.message = message


class InvalidTokenError(Exception):
    """Exception raised for invalid token."""
    def __init__(self, message):
        super().__init__()
        self.message = message


class Event:
    """Class representing a single MIDI event.

    NOTE_ON and NOTE_OFF events are supported.
    An event consists of program, note, velocity, and time.
    The NOTE_ON event always has positive velocity.
    The NOTE_OFF event always has velocity 0.
    Time is the delta tick divided by ticks_per_beat.
    """
    def __init__(self, program: int, note: int, velocity: int, time: Fraction):
        self.program = program
        self.note = note
        self.velocity = velocity
        self.time = time

    def __repr__(self):
        return f"Event(program={self.program}, note={self.note}, velocity={self.velocity}, time={self.time})"

    def __str__(self):
        event_type = "NOTE_ON" if self.velocity > 0 else "NOTE_OFF"
        return f"{event_type} program={self.program}, note={self.note}, velocity={self.velocity}, time={self.time}"

    def __eq__(self, __o: object) -> bool:
        if not isinstance(__o, Event):
            return False
        return self.program == __o.program and self.note == __o.note \
            and self.velocity == __o.velocity and self.time == __o.time


class Converter:
    """Converts a list of Events to an numpy array of tokens and vice versa.

    A token is a representation of an Event with the following columns:
    - program
    - note
    - velocity
    - time_num
    - time_denum
    Where time_num and time_denum are the numerator and denominator of the delta
    time of the event.

    The delta time is simplified to the nearest fraction in which the numerator
    or the denominator is less than UINT8_MAX."""
    def __init__(self, token_type: np.dtype):
        self.type = token_type

    def events_to_tokens(self, events: List[Event]) -> ndarray:
        """Converts a list of Events to a numpy array of tokens.

        Args:
            events: List of Events (required).

        Returns:
            A numpy array of tokens."""

        # Initialize list of tokens
        tokens = []
        # simple_limit is the maximum that can be represented by self.type
        simple_limit = np.iinfo(self.type).max
        # Initialize current time
        current_time = Fraction(0)
        # Iterate through events
        for event in events:
            # Calculate delta time
            delta_time = event.time - current_time
            if delta_time > simple_limit:
                delta_time = Fraction(simple_limit)
            # Calculate time numerator and denominator
            time_num = delta_time.numerator
            time_denum = delta_time.denominator
            # Simplify time
            interval = 0
            # If numerator > denominator, then numerator must be less than simple_limit
            # Therefore interval is inverse of what the maximum denominator can be
            # i.e. floor(denominator * simple_limit / numerator)
            if time_num > time_denum and time_num > simple_limit:
                interval = 1 / np.floor(time_denum * simple_limit / time_num)
            # If denominator > numerator, then denominator must be less than simple_limit
            # Therefore interval is simply inverse of simple_limit
            elif time_denum > time_num and time_denum > simple_limit:
                interval = 1 / simple_limit

            # If interval is positive, simplify time by calling simplify(lower, upper)
            # Where lower = time - interval / 2 and upper = time + interval / 2
            if interval > 0:
                time = simplify(delta_time - interval / 2,
                                delta_time + interval / 2)
                # Update numerator & denominator
                time_num = time.numerator
                time_denum = time.denominator

            # Append token to list of tokens
            tokens.append([
                event.program, event.note, event.velocity, time_num, time_denum
            ])
            # Update current time
            current_time = event.time

        assert np.array(tokens).max(
        ) <= simple_limit, f"{np.array(tokens).max()} > {simple_limit}"
        # Convert list of tokens to numpy array
        return np.array(tokens, dtype=self.type)

    def tokens_to_events(self, tokens: ndarray) -> List[Event]:
        """Converts a numpy array of tokens to a list of Events.

        Args:
            tokens: numpy array of tokens (required).

        Returns:
            A list of Events."""

        # Initialize list of events
        events = []
        # Initialize current time
        current_time = Fraction(0)
        # Iterate through tokens
        for token in tokens:
            # Parse token
            program, note, velocity, time_num, time_denum = token
            # Update current time
            current_time += Fraction(time_num, time_denum)
            # Append event to list of events
            events.append(Event(program, note, velocity, current_time))

        # Return list of events
        return events


class Modifier:
    """Flattens, unflattens, and augments tokens."""
    def __init__(self, num_special: int, num_program: int, num_note: int,
                 num_velocity: int, num_time_num: int, note_shift: int,
                 velocity_scale: float):
        self.program_offset = num_special
        self.note_offset = self.program_offset + num_program
        self.velocity_offset = self.note_offset + num_note
        self.time_num_offset = self.velocity_offset + num_velocity
        self.time_denum_offset = self.time_num_offset + num_time_num
        self.note_max = num_note - 1
        self.velocity_max = num_velocity - 1
        self.note_shift = note_shift
        self.velocity_scale = velocity_scale
        self.rng = np.random.default_rng()
        self.pad = 0
        self.begin = 1
        self.end = 2

    def augment(self, tokens: ndarray) -> ndarray:
        """Augment a 2d numpy array of tokens.

        A token consists of five columns:
        [program, note, velocity, time_num, time_denum],
        where time_num and time_denum are the numerator and denominator of the
        delta time.
        We randomly sample a note offset from a integer range of [-self.note_shift,
        self.note_shift].
        We randomly sample a velocity scale from a log-uniform distribution of
        [1/self.velocity_scale, self.velocity_scale].
        We currently do not augment time.

        Args:
            tokens: Numpy array of tokens to augment (required).

        Returns:
            Augmented tokens."""
        # Sample note shift from integer uniform distribution in range [-note_shift,
        # note_shift].
        note_shift = self.rng.integers(-self.note_shift,
                                       self.note_shift,
                                       endpoint=True)
        # Sample velocity scale from log-uniform distribution in range
        # [1/velocity_scale, velocity_scale].
        log_velocity_scale = np.log(self.velocity_scale)
        velocity_scale = np.exp(
            self.rng.uniform(
                low=-log_velocity_scale,
                high=log_velocity_scale,
                size=1,
            ))
        # Shift notes and clip to [0, note_max].
        tokens[:, 1] = np.clip(tokens[:, 1] + note_shift, 0, self.note_max)
        # Scale velocities, round to nearest integer, and clip to [0, velocity_max].
        tokens[:, 2] = np.clip(np.rint(tokens[:, 2] * velocity_scale), 0,
                               self.velocity_max)
        return tokens

    def flatten(self, tokens: ndarray) -> ndarray:
        """Flattens a 2d numpy array of tokens.

        See augment() for 2d token format.

        Zero velocity tokens are removed.
        Zero time tokens are removed.

        Args:
            tokens: 2d numpy array of tokens (required).

        Returns:
            A flattened numpy array of tokens."""

        # Initialize flattened tokens
        flattened_tokens = []
        # Add offsets to tokens
        tokens[:, 0] += self.program_offset
        tokens[:, 1] += self.note_offset
        tokens[:, 2] += self.velocity_offset
        tokens[:, 3] += self.time_num_offset
        tokens[:, 4] += self.time_denum_offset
        # Iterate through tokens
        for token in tokens:
            program, note, velocity, time_num, time_denum = token
            # Append program, note
            flattened_tokens.append(program)
            flattened_tokens.append(note)
            # If velocity is not zero, append velocity
            if velocity > self.velocity_offset:
                flattened_tokens.append(velocity)
            # If time is not zero, append time numerator and denominator
            if time_num > self.time_num_offset:
                flattened_tokens.append(time_num)
                flattened_tokens.append(time_denum)
        # Convert flattened tokens to numpy array
        return np.array(flattened_tokens, dtype=np.int64)

    def unflatten(self, tokens: ndarray) -> ndarray:
        """Unflattens a 1d numpy array of tokens.

        See flatten() for 1d token format.

        Args:
            tokens: 1d numpy array of tokens (required).

        Returns:
            A 2d numpy array of tokens."""

        # Initialize unflattened tokens
        unflattened_tokens = []
        program = 0
        # Iterate through tokens
        for token in tokens:
            if self.program_offset <= token < self.note_offset:
                # Set program
                program = token - self.program_offset
            elif self.note_offset <= token < self.velocity_offset:
                # Append program, note
                unflattened_tokens.append(
                    [program, token - self.note_offset, 0, 0, 1])
            elif self.velocity_offset <= token < self.time_num_offset:
                # Update token if tokens exist
                if unflattened_tokens:
                    unflattened_tokens[-1][2] = token - self.velocity_offset
            elif self.time_num_offset <= token < self.time_denum_offset:
                # Update token if tokens exist
                if unflattened_tokens:
                    unflattened_tokens[-1][3] = token - self.time_num_offset
            elif self.time_denum_offset <= token:
                # Update token if tokens exist
                if unflattened_tokens:
                    unflattened_tokens[-1][4] = token - self.time_denum_offset

        # Convert unflattened tokens to numpy array
        return np.array(unflattened_tokens, dtype=np.uint8)

    def pad_or_slice(self, tokens: ndarray, length: int) -> ndarray:
        """Pads a 1d numpy array of tokens.

        If the length of the tokens is less than the given length,
        the tokens are padded with PAD.
        If the length of the tokens is greater than the given length,
        a random slice of the tokens is returned.

        Args:
            tokens: 1d numpy array of tokens (required).
            length: Length of padded tokens (required).

        Returns:
            A 1d numpy array of tokens of length `length`."""

        # Pad begin & end tokens
        tokens = np.concatenate([[self.begin], tokens, [self.end]])
        # If length is greater than tokens length, pad tokens
        if length > len(tokens):
            tokens = np.concatenate(
                [tokens, [self.pad] * (length - len(tokens))])
        # If length is less than tokens length, randomly slice tokens
        elif length < len(tokens):
            start_index = self.rng.integers(0,
                                            len(tokens) - length,
                                            endpoint=True)
            tokens = tokens[start_index:start_index + length]
        # Return padded tokens
        return tokens


def process_tokens(data_dir: Path, text_dir: Path, process_dir: Path) -> None:
    """Converts all MIDI files in data_dir into tokens and saves then as UINT8
    .npy files in process_dir.

    The .npy files have the same file structure as the original MIDI files.
    The .npy files are named after the original MIDI files.
    If process_dir already exists, the function does nothing.

    If EOFError, KeySignatureError, or IndexError is raised, the exception name
    and the file name are printed and the function continues.

    All successfully converted midi filenames are sorted and stored in `midi.txt`
    under text_dir.

    Args:
        data_dir: Path to directory containing MIDI files (required).
        text_dir: Path to directory to store `midi.txt` (required).
        process_dir: Path to directory to store .npy files (required).
    """

    # Check if process_dir exists.
    if process_dir.exists():
        return

    # Create process_dir.
    os.makedirs(process_dir)
    # Initialize list of successfully converted midi filenames.
    filenames = []
    # Type is UINT8.
    token_type = np.uint8
    converter = Converter(token_type=token_type)

    # Recursively iterate over all MIDI files in data_dir.
    for path in tqdm(data_dir.rglob("*.mid")):
        # Type hint path
        path: Path
        # Get relative path of MIDI file from data_dir.
        midi_rel_path = path.relative_to(data_dir)
        # Get path of .npy file in process_dir.
        npy_path = process_dir / midi_rel_path.with_suffix(".npy")
        # Create parent directories of .npy file if necessary.
        npy_path.parent.mkdir(parents=True, exist_ok=True)
        # Read MIDI file with clip=True and check for errors.
        try:
            midi_file = MidiFile(path, clip=True)
        except (EOFError, KeySignatureError, IndexError) as exception:
            tqdm.write(f"{midi_rel_path}: {exception.__class__.__name__}")
            continue
        # Convert MIDI file to events.
        events = midi_to_events(midi_file)
        # Convert events to tokens.
        tokens = converter.events_to_tokens(events)
        # Check if tokens exceed the maximum value of token_type.
        if np.any(tokens > np.iinfo(token_type).max):
            tqdm.write(f"{midi_rel_path}: Token exceeds maximum value.")
            continue
        # Save tokens to .npy file.
        np.save(npy_path, tokens)
        # Add filename to list of filenames.
        filenames.append(str(midi_rel_path) + "\n")

    # Sort filenames.
    filenames.sort()
    # Write filenames to file.
    with open(text_dir / "midi.txt", mode="w", encoding="utf-8") as file:
        file.writelines(filenames)


def midi_to_events(midi_file: MidiFile) -> List[Event]:
    """Converts a MIDI file to a list of Events.

    A MIDI file is a list of tracks, where each track is a list of messages.
    There are three types of messages to consider:
    - NOTE_ON
    - NOTE_OFF
    - PROGRAM_CHANGE
    The PROGRAM_CHANGE message is used to change the instrument of a channel.
    Channel 10 is reserved for percussion. (However, mido numbers channels from
    0 to 15, so channel 10 is actually channel 9.)

    The NOTE_ON message specifies the channel, note, velocity, and start time.
    The NOTE_OFF message specifies the channel, note, and end time.
    Note that a NOTE_ON message with velocity 0 is equivalent to a NOTE_OFF
    message.

    Since messages need to be chronologically ordered and tracks run simultaneously,
    we first calculate the timing of all messages in each track.
    Then, we concatenate all tracks and sort the messages chronologically.
    We then iterate through the messages and create Events.

    Args:
        midi_file: MIDI file to convert (required).

    Returns:
        A list of Events."""

    # Initialize list of events.
    events: List[Event] = []
    # Initialize channel-to-program conversion list.
    programs = [0] * 16
    # Channel 9 is reserved for percussion.
    programs[9] = 128
    # Initialize list of messages.
    messages: List[Message] = []
    # Get ticks per beat.
    ticks_per_beat = midi_file.ticks_per_beat

    # Iterate through tracks.
    for track in midi_file.tracks:
        # Initialize current time.
        current_time = 0
        # Iterate through messages in track.
        for message in track:
            # Calculate current time.
            current_time += message.time
            # Update message time.
            message.time = current_time
            # Append message to list of messages.
            messages.append(message)

    # Sort messages chronologically with attribute `time`.
    messages.sort(key=lambda message: message.time)
    # Iterate through messages.
    for message in messages:
        # Check if message is a NOTE_ON message.
        if message.type == "note_on" and message.velocity > 0:
            # Get program.
            program = programs[message.channel]
            # Create event.
            event = Event(program, message.note, message.velocity,
                          Fraction(message.time, ticks_per_beat))
            # Append event to list of events.
            events.append(event)
        # Check if message is a NOTE_OFF message.
        elif message.type == "note_off" or (message.type == "note_on"
                                            and message.velocity == 0):
            # Get program.
            program = programs[message.channel]
            # Create event.
            event = Event(program, message.note, 0,
                          Fraction(message.time, ticks_per_beat))
            # Append event to list of events.
            events.append(event)
        # Check if message is a PROGRAM_CHANGE message.
        elif message.type == "program_change":
            # Update channel-to-program conversion list.
            programs[message.channel] = message.program

    # Return list of events.
    return events


def events_to_midi(events: List[Event]) -> MidiFile:
    """Converts a list of Events to a MIDI file.

    Currently assigns each instrument to a single channel.
    A MIDI file supports up to 16 channels.
    Therefore, more than 15 non-percussion instruments will not be supported.

    See `midi_to_events` for more information on the format of Events.

    Args:
        events: List of Events to convert (required).

    Returns:
        A MIDI file."""

    # Default ticks per beat.
    ticks_per_beat = 120
    # Initialize MIDI file.
    midi_file = MidiFile(ticks_per_beat=ticks_per_beat)
    # Get set of unique programs.
    programs = set(event.program for event in events)
    # Check if more than 15 non-percussion instruments are used.
    if 128 in programs:
        programs.remove(128)
    if len(programs) > 15:
        raise ValueError("More than 15 non-percussion instruments are used.")
    # Define program-to-channel conversion map.
    # Start from channel 10, since channel 9 is reserved for percussion.
    channel_map = {
        program: channel % 16
        for channel, program in enumerate(programs, start=10)
    }
    channel_map[128] = 9

    # Add a track for each channel.
    for _ in range(16):
        midi_file.tracks.append(MidiTrack())
    # Set program for each channel.
    for program, channel in channel_map.items():
        track = midi_file.tracks[channel]
        # If percussion, set program to 0.
        if channel == 9:
            program = 0
        track.append(
            Message("program_change", program=program, channel=channel,
                    time=0))

    # Iterate through events.
    for event in events:
        # Get channel.
        channel = channel_map[event.program]
        # Get track.
        track: MidiTrack = midi_file.tracks[channel]
        # Check if event is a NOTE_ON event.
        if event.velocity > 0:
            # Create NOTE_ON message.
            message = Message("note_on",
                              channel=channel,
                              note=event.note,
                              velocity=event.velocity,
                              time=round(event.time * ticks_per_beat))
            # Append message to track.
            track.append(message)
        # Check if event is a NOTE_OFF event.
        elif event.velocity == 0:
            # Create NOTE_OFF message.
            message = Message("note_off",
                              channel=channel,
                              note=event.note,
                              velocity=0,
                              time=round(event.time * ticks_per_beat))
            # Append message to track.
            track.append(message)

    # Return MIDI file.
    return midi_file


def simplify(lower: float, upper: float) -> Fraction:
    """Finds the simplest fraction between lower and upper.

    The simplest fraction is defined as the fraction with the smallest denominator
    that is greater than or equal to lower and less than or equal to upper.

    The algorithm used is as follows:
    - If 0 exists in the interval, return 0.
    - If 1 exists in the interval, return 1.
    - If 0 < lower < upper < 1, return the inverse of the simplified fraction of
      the inverted interval.
    - If 1 < lower, simplify(lower, upper) = q + simplify(lower - q, upper - q) where
      q is the largest integer smaller than lower.

    Args:
        lower: The lower bound of the interval (required).
        upper: The upper bound of the interval (required).

    Returns:
        The simplest fraction between lower and upper."""

    # Check if 0 exists in the interval.
    if lower <= 0 <= upper:
        return Fraction(0)

    # Check if 1 exists in the interval.
    if lower <= 1 <= upper:
        return Fraction(1)

    # Check if 0 < lower < upper < 1.
    if upper < 1:
        return 1 / simplify(1 / upper, 1 / lower)

    # Find the largest integer less than or equal to lower.
    integer = int(lower)
    # Subtract that integer from the interval.
    lower -= integer
    upper -= integer
    # Return the simplified fraction of the new interval added to the integer.
    return simplify(lower, upper) + integer
