from enum import IntEnum
from operator import attrgetter, itemgetter
from pathlib import Path
from typing import List, Optional

import numpy as np
from mido import Message, MidiFile
from mido.midifiles.meta import KeySignatureError
from numpy import ndarray
from tqdm import tqdm

from config.config import CustomConfig


class InvalidProgramError(Exception):
    pass


class InvalidTypeError(Exception):
    pass


class InvalidNoteError(Exception):
    pass


class InvalidTickError(Exception):
    pass


class InvalidTokenError(Exception):
    pass


class MessageType(IntEnum):
    NOTE_OFF = 1
    NOTE_ON = 2


class Event:
    def __init__(
        self,
        message_type: Optional[MessageType],
        tick: int,
        program: int,
        note: Optional[int] = None,
    ):
        self.type = message_type
        self.tick = tick
        self.program = program
        self.note = note


class Tokenizer:
    def __init__(self, cfg: CustomConfig) -> None:
        self.pad = 0
        self.begin = 1
        self.end = 2
        self.note_off = 3
        self.note_on = 4
        self.num_program = cfg.num_program
        self.num_note = cfg.num_note
        self.num_tick = cfg.num_tick
        self.special_limit = cfg.num_special
        self.program_limit = self.special_limit + self.num_program
        self.note_limit = self.program_limit + self.num_note
        self.tick_limit = self.note_limit + self.num_tick

    def program_to_token(self, program: int) -> int:
        if 0 > program or program >= self.num_program:
            raise InvalidProgramError
        return program + self.special_limit

    def type_to_token(self, message_type: MessageType) -> int:
        if message_type == MessageType.NOTE_ON:
            return self.note_on
        if message_type == MessageType.NOTE_OFF:
            return self.note_off
        raise InvalidTypeError

    def note_to_token(self, note: int) -> int:
        if 0 > note or note >= self.num_note:
            raise InvalidNoteError
        return note + self.program_limit

    def tick_to_token(self, tick: int) -> int:
        if 0 > tick or tick >= self.num_tick:
            raise InvalidTickError
        return tick + self.note_limit

    def tokenize(self, event_list: List[Event]) -> ndarray:
        token_list = [self.begin]
        prev_tick = 0
        prev_program = -1
        prev_on_off = MessageType.NOTE_OFF
        for event in event_list:
            tick_delta = min(event.tick - prev_tick, self.num_tick)
            if tick_delta > 0:
                token_list.append(self.tick_to_token(tick_delta - 1))
                prev_tick = event.tick
            if event.program != prev_program:
                token_list.append(self.program_to_token(event.program))
                prev_program = event.program
            if event.type != prev_on_off:
                token_list.append(self.type_to_token(event.type))
                prev_on_off = event.type
            if event.note is not None:
                token_list.append(self.note_to_token(event.note))
        token_list.append(self.end)

        return np.array(token_list, dtype=np.int64)

    def tokens_to_events(self, tokens: ndarray) -> List[Event]:
        event_list: List[Event] = []
        cur_tick = 0
        cur_program = -1
        cur_on_off = MessageType.NOTE_OFF
        for token in tokens:
            if token < self.special_limit:
                if token == self.end:
                    break
                if token == self.note_on:
                    cur_on_off = MessageType.NOTE_ON
                if token == self.note_off:
                    cur_on_off = MessageType.NOTE_OFF
                continue
            if token < self.program_limit:
                cur_program = token - self.special_limit
            elif token < self.note_limit:
                note = token - self.program_limit
                event_list.append(
                    Event(
                        message_type=cur_on_off,
                        tick=cur_tick,
                        program=cur_program,
                        note=note,
                    )
                )
            elif token < self.tick_limit:
                tick = token - self.note_limit + 1
                cur_tick += tick
            else:
                raise InvalidTokenError

        return event_list


def prepare_data(cfg: CustomConfig) -> None:
    if not cfg.process_dir.is_dir():
        cfg.process_dir.mkdir(parents=True)
    filenames = []
    tokenizer = Tokenizer(cfg)
    for path in tqdm(cfg.data_dir.glob("**/*.mid")):
        path: Path
        relative_path = path.relative_to(cfg.data_dir)
        try:
            filename = cfg.process_dir / relative_path.with_suffix(".npy")
            if filename.exists():
                continue
            filename.parent.mkdir(parents=True, exist_ok=True)
            event_list = read_midi(MidiFile(filename=path, clip=True))
            tokens = tokenizer.tokenize(event_list)
            np.save(filename, tokens.astype(np.int16))
            filenames.append(str(relative_path) + "\n")
        except (EOFError, KeySignatureError, IndexError) as exception:
            tqdm.write(f"{type(exception).__name__}: {relative_path}")
            continue

    filenames.sort()
    text_path = cfg.file_dir / "midi.txt"
    if not text_path.exists():
        with open(text_path, mode="w", encoding="utf-8") as file:
            file.writelines(filenames)


def read_midi(midi_file: MidiFile) -> List[Event]:
    events: List[Event] = []
    messages = []
    programs = np.zeros(16, dtype=np.int64)
    ticks_per_beat = midi_file.ticks_per_beat

    for track in midi_file.tracks:
        cur_tick = 0
        for message in track:
            cur_tick += message.time
            messages.append((round(cur_tick / ticks_per_beat * 30), message))

    messages.sort(key=itemgetter(0))

    for tick, message in messages:
        if message.type == "note_on" or message.type == "note_off":
            if message.channel == 9:
                program = message.note + 128
                note = None
            else:
                program = programs[message.channel]
                note = message.note
            events.append(
                Event(
                    message_type=MessageType.NOTE_ON
                    if message.type == "note_on" and message.velocity > 0
                    else MessageType.NOTE_OFF,
                    tick=tick,
                    program=program,
                    note=note,
                )
            )
        elif message.type == "program_change" and message.channel != 9:
            programs[message.channel] = message.program

    events.sort(key=attrgetter("tick", "type", "program", "note"))

    return events


def write_midi(event_list: List[Event]) -> MidiFile:
    midi_file = MidiFile()
    track = midi_file.add_track()
    prev_tick = 0
    program_set = set()
    ticks_per_beat = midi_file.ticks_per_beat
    for event in event_list:
        if event.program < 128:
            program_set.add(event.program)

    if len(program_set) > 15:
        print("More than 15 non-drum instruments not supported")
        return midi_file

    program_map = {}
    for i, program in enumerate(program_set):
        program_map[program] = (i + 10) % 16

    for track in midi_file.tracks:
        for program, channel in program_map.items():
            track.append(Message("program_change", channel=channel, program=program, time=0))

    for event in event_list:
        if event.program < 128:
            channel = program_map[event.program]
            note = event.note
        else:
            channel = 9
            note = event.program - 128
        track.append(
            Message(
                "note_on" if event.type == MessageType.NOTE_ON else "note_off",
                channel=channel,
                note=note,
                velocity=64 if event.type == MessageType.NOTE_ON else 0,
                time=(event.tick - prev_tick) * ticks_per_beat / 30,
            )
        )
        prev_tick = event.tick

    return midi_file
