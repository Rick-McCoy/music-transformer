from enum import IntEnum
from operator import itemgetter
from typing import Iterable, List, Optional

import numpy as np
from mido import Message, MidiFile
from numpy import ndarray

from config.config import (
    BEGIN,
    DRUM_LIMIT,
    END,
    NOTE_LIMIT,
    NOTE_OFF,
    NOTE_ON,
    NUM_DRUM,
    NUM_NOTE,
    NUM_PROGRAM,
    NUM_TICK,
    PAD,
    PROGRAM_LIMIT,
    SPECIAL_LIMIT,
    TICK_LIMIT,
    TIE,
)


class MessageType(IntEnum):
    NOTE_OFF = 1
    NOTE_ON = 2


class InvalidProgramError(Exception):
    def __init__(self, program: int):
        super().__init__()
        self.program = program

    def __repr__(self):
        return f"Invalid program: {self.program}"

    def __str__(self):
        return self.__repr__()


class InvalidDrumError(Exception):
    def __init__(self, drum: int):
        super().__init__()
        self.drum = drum

    def __repr__(self):
        return f"Invalid drum: {self.drum}"

    def __str__(self):
        return self.__repr__()


class InvalidTypeError(Exception):
    def __init__(self, message_type: MessageType):
        super().__init__()
        self.message_type = message_type

    def __repr__(self):
        return f"Invalid type: {self.message_type}"

    def __str__(self):
        return self.__repr__()


class InvalidNoteError(Exception):
    def __init__(self, note: int):
        super().__init__()
        self.note = note

    def __repr__(self):
        return f"Invalid note: {self.note}"

    def __str__(self):
        return self.__repr__()


class InvalidTickError(Exception):
    def __init__(self, tick: int):
        super().__init__()
        self.tick = tick

    def __repr__(self):
        return f"Invalid tick: {self.tick}"

    def __str__(self):
        return self.__repr__()


class InvalidTokenError(Exception):
    def __init__(self, token: int):
        super().__init__()
        self.token = token

    def __repr__(self):
        return f"Invalid token: {self.token}"

    def __str__(self):
        return self.__repr__()


class Event:
    def __init__(
        self,
        message_type: Optional[MessageType],
        tick: int,
        program: Optional[int] = None,
        note: Optional[int] = None,
        drum: Optional[int] = None,
    ):
        self.type = message_type
        self.tick = tick
        self.program = program
        self.note = note
        self.drum = drum

    def __repr__(self):
        string = "<" + " ".join(f"{k}={v}" for k, v in self.__dict__.items()) + ">"
        return string

    def __str__(self):
        return self.__repr__()


def program_to_token(program: int) -> int:
    if 0 <= program < NUM_PROGRAM:
        return program + SPECIAL_LIMIT
    raise InvalidProgramError(program)


def drum_to_token(drum: int) -> int:
    if 0 <= drum < NUM_DRUM:
        return drum + PROGRAM_LIMIT
    raise InvalidDrumError(drum)


def type_to_token(message_type: MessageType) -> int:
    if message_type == MessageType.NOTE_ON:
        return NOTE_ON
    if message_type == MessageType.NOTE_OFF:
        return NOTE_OFF
    raise InvalidTypeError(message_type)


def note_to_token(note: int) -> int:
    if 0 <= note < NUM_NOTE:
        return note + DRUM_LIMIT
    raise InvalidNoteError(note)


def tick_to_token(tick: int) -> int:
    if 0 <= tick < NUM_TICK:
        return tick + NOTE_LIMIT
    raise InvalidTickError(tick)


def token_to_string(token: int) -> str:
    if token < SPECIAL_LIMIT:
        if token == PAD:
            return "PAD"
        if token == BEGIN:
            return "BEGIN"
        if token == END:
            return "END"
        if token == NOTE_OFF:
            return "NOTE_OFF"
        if token == NOTE_ON:
            return "NOTE_ON"
        if token == TIE:
            return "TIE"
        raise InvalidTokenError(token)
    if token < PROGRAM_LIMIT:
        return f"PROG_{token - SPECIAL_LIMIT:03d}"
    if token < DRUM_LIMIT:
        return f"DRUM_{token - PROGRAM_LIMIT:03d}"
    if token < NOTE_LIMIT:
        return f"NOTE_{token - DRUM_LIMIT:03d}"
    if token < TICK_LIMIT:
        return f"TICK_{token - NOTE_LIMIT + 1:03d}"
    raise InvalidTokenError(token)


def tokens_to_string(tokens: Iterable[int]) -> str:
    result = ""
    line = ["", "", "", ""]

    def update_result(result: str, line: List[str], string: str, index: int):
        if line[index]:
            result += "\n| " + " | ".join(f"{s:8s}" for s in line) + " |"
            line = ["", "", "", ""]
        line[index] = string
        return result, line

    for token in tokens:
        string = token_to_string(token)
        if token < SPECIAL_LIMIT:
            if token in (PAD, BEGIN, END, TIE):
                if any(line):
                    result += "\n| " + " | ".join(f"{s:8s}" for s in line) + " |"
                line = [string, "", "", ""]
            elif token in (NOTE_ON, NOTE_OFF):
                result, line = update_result(result, line, string, 1)
            else:
                raise InvalidTokenError(token)
        elif token < PROGRAM_LIMIT:
            result, line = update_result(result, line, string, 2)
        elif token < NOTE_LIMIT:
            result, line = update_result(result, line, string, 3)
        elif token < TICK_LIMIT:
            result, line = update_result(result, line, string, 0)
        else:
            raise InvalidTokenError(token)

    if any(line):
        result += f"\n| {line[0]:8s} | {str(line[1]):8s} | {line[2]:8s} | {line[3]:8s} |"
    return result


def events_to_tokens(event_list: List[Event]) -> ndarray:
    token_list = [BEGIN]
    prev_tick = 0
    prev_program = None
    prev_on_off = MessageType.NOTE_OFF
    for event in event_list:
        tick_delta = event.tick - prev_tick
        while tick_delta > 0:
            token_list.append(tick_to_token(min(tick_delta, NUM_TICK) - 1))
            tick_delta -= NUM_TICK
        prev_tick = event.tick
        if event.type is not None and event.type != prev_on_off:
            token_list.append(type_to_token(event.type))
            prev_on_off = event.type
        if event.program is not None and event.program != prev_program:
            token_list.append(program_to_token(event.program))
            prev_program = event.program
        if event.note is not None:
            token_list.append(note_to_token(event.note))
        if event.drum is not None:
            token_list.append(drum_to_token(event.drum))
    token_list.append(END)

    return np.array(token_list, dtype=np.int64)


def tokens_to_events(tokens: ndarray) -> List[Event]:
    event_list: List[Event] = []
    cur_tick = 0
    cur_program = 0
    cur_on_off = MessageType.NOTE_OFF
    if TIE in tokens:
        tie_index = np.where(tokens == TIE)[0][0]
        tokens = tokens[tie_index + 1 :]
    for token in tokens:
        if token < SPECIAL_LIMIT:
            if token == END:
                break
            if token == NOTE_ON:
                cur_on_off = MessageType.NOTE_ON
            if token == NOTE_OFF:
                cur_on_off = MessageType.NOTE_OFF
        elif token < PROGRAM_LIMIT:
            cur_program = token - SPECIAL_LIMIT
        elif token < DRUM_LIMIT:
            drum = token - PROGRAM_LIMIT
            event_list.append(
                Event(
                    message_type=cur_on_off,
                    tick=cur_tick,
                    drum=drum,
                )
            )
        elif token < NOTE_LIMIT:
            note = token - DRUM_LIMIT
            event_list.append(
                Event(
                    message_type=cur_on_off,
                    tick=cur_tick,
                    program=cur_program,
                    note=note,
                )
            )
        elif token < TICK_LIMIT:
            tick = token - NOTE_LIMIT + 1
            cur_tick += tick
        else:
            raise InvalidTokenError(token)

    return event_list


def augment_tokens(tokens: ndarray) -> ndarray:
    note_tokens = tokens[(DRUM_LIMIT <= tokens) & (tokens < NOTE_LIMIT)]
    note_shift = np.random.randint(-12, 12)
    note_tokens = np.clip(note_tokens + note_shift, DRUM_LIMIT, NOTE_LIMIT - 1)
    tokens[(DRUM_LIMIT <= tokens) & (tokens < NOTE_LIMIT)] = note_tokens
    return tokens


def determine_on_notes(tokens: ndarray) -> ndarray:
    programs = np.zeros(NUM_PROGRAM, dtype=bool)
    programs[tokens[(tokens >= SPECIAL_LIMIT) & (tokens < PROGRAM_LIMIT)] - SPECIAL_LIMIT] = True
    on_notes = np.zeros((NUM_PROGRAM, NUM_NOTE), dtype=bool)
    on_drums = np.zeros((NUM_DRUM,), dtype=bool)
    on_drums[tokens[(tokens >= PROGRAM_LIMIT) & (tokens < DRUM_LIMIT)] - PROGRAM_LIMIT] = True

    program = 0
    for token in tokens:
        if SPECIAL_LIMIT <= token < PROGRAM_LIMIT:
            program = token - SPECIAL_LIMIT
        elif DRUM_LIMIT <= token < NOTE_LIMIT:
            note = token - DRUM_LIMIT
            on_notes[program, note] = ~on_notes[program, note]

    result: List[int] = []
    for program in np.where(programs)[0]:
        result.append(program_to_token(program))
        for note in np.where(on_notes[program])[0]:
            result.append(note_to_token(note))

    for drum in np.where(on_drums)[0]:
        result.append(drum_to_token(drum))

    result.append(TIE)

    return np.array(result, dtype=np.int64)


def read_midi(midi_file: MidiFile) -> List[Event]:
    events: List[Event] = []
    messages = []
    programs = [0 for _ in range(16)]
    ticks_per_beat = midi_file.ticks_per_beat

    for track in midi_file.tracks:
        cur_tick = 0
        for message in track:
            cur_tick += message.time
            messages.append((round(cur_tick / ticks_per_beat * 120), message))

    messages.sort(key=itemgetter(0))

    for tick, message in messages:
        if message.type == "note_on" or message.type == "note_off":
            if message.channel == 9:
                if message.type == "note_off" or message.velocity == 0:
                    continue
                program = None
                note = None
                drum = message.note
            else:
                program = programs[message.channel]
                note = message.note
                drum = None
            events.append(
                Event(
                    message_type=MessageType.NOTE_ON
                    if message.type == "note_on" and message.velocity > 0
                    else MessageType.NOTE_OFF,
                    tick=tick,
                    program=program,
                    note=note,
                    drum=drum,
                )
            )
        elif message.type == "program_change" and message.channel != 9:
            programs[message.channel] = message.program

    events.sort(
        key=lambda event: (
            event.tick,
            event.type,
            event.program is None,
            event.program,
            event.note is None,
            event.note,
            event.drum is None,
            event.drum,
        )
    )

    return events


def write_midi(event_list: List[Event]) -> MidiFile:
    midi_file = MidiFile()
    track = midi_file.add_track()
    prev_tick = 0
    program_set = set()
    ticks_per_beat = midi_file.ticks_per_beat
    for event in event_list:
        if event.program is not None:
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
        if event.program is not None:
            channel = program_map[event.program]
            note = event.note
        else:
            channel = 9
            note = event.drum
        track.append(
            Message(
                "note_on" if event.type == MessageType.NOTE_ON else "note_off",
                channel=channel,
                note=note,
                velocity=64 if event.type == MessageType.NOTE_ON else 0,
                time=round((event.tick - prev_tick) * ticks_per_beat / 120),
            )
        )
        prev_tick = event.tick

    return midi_file
