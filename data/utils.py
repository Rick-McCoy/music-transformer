from enum import IntEnum
import glob
from operator import attrgetter, itemgetter
import os
from pathlib import Path
from typing import List, Optional, Tuple

from hydra.utils import to_absolute_path
from mido import MidiFile
from mido.messages.messages import Message
from mido.midifiles.meta import KeySignatureError
import numpy as np
from numpy import ndarray
from omegaconf.dictconfig import DictConfig
from tqdm import tqdm


class InvalidProgramError(Exception):
    pass


class InvalidTypeError(Exception):
    pass


class InvalidNoteError(Exception):
    pass


class InvalidVelocityError(Exception):
    pass


class InvalidControlError(Exception):
    pass


class InvalidValueError(Exception):
    pass


class InvalidPitchError(Exception):
    pass


class InvalidTickError(Exception):
    pass


class InvalidTokenError(Exception):
    pass


class MessageType(IntEnum):
    NOTE_OFF = 1
    NOTE_ON = 2
    POLYTOUCH = 3
    CONTROL_CHANGE = 4
    PROGRAM_CHANGE = 5
    AFTERTOUCH = 6
    PITCHWHEEL = 7
    SYSEX = 8
    QUARTER_FRAME = 9
    SONGPOS = 10
    SONG_SELECT = 11
    TUNE_REQUEST = 12
    CLOCK = 13
    START = 14
    CONTINUE = 15
    STOP = 16
    ACTIVE_SENSING = 17
    RESET = 18


class Event:
    def __init__(self,
                 message_type: MessageType,
                 tick: int,
                 program: int,
                 note: Optional[int] = None,
                 velocity: Optional[int] = None,
                 control: Optional[int] = None,
                 value: Optional[int] = None,
                 pitch: Optional[int] = None):
        self.type = message_type
        self.tick = tick
        self.program = program
        self.note = note
        self.velocity = velocity
        self.control = control
        self.value = value
        self.pitch = pitch


class Tokenizer:
    def __init__(self, cfg: DictConfig) -> None:
        self.pad = 0
        self.begin = 1
        self.end = 2
        self.num_program = cfg.model.num_program
        self.num_type = cfg.model.num_type
        self.num_program_type = self.num_program * self.num_type
        self.num_note = cfg.model.num_note
        self.num_velocity = cfg.model.num_velocity
        self.num_control = cfg.model.num_control
        self.num_value = cfg.model.num_value
        self.num_pitch_1 = cfg.model.num_pitch_1
        self.num_pitch_2 = cfg.model.num_pitch_2
        self.num_tick = cfg.model.num_tick
        self.special_limit = 3
        self.program_type_limit = self.special_limit + self.num_program_type
        self.note_limit = self.program_type_limit + self.num_note
        self.velocity_limit = self.note_limit + self.num_velocity
        self.control_limit = self.velocity_limit + self.num_control
        self.value_limit = self.control_limit + self.num_value
        self.pitch_1_limit = self.value_limit + self.num_pitch_1
        self.pitch_2_limit = self.pitch_1_limit + self.num_pitch_2
        self.tick_limit = self.pitch_2_limit + self.num_tick
        self.type_map = {
            MessageType.NOTE_OFF: 0,
            MessageType.NOTE_ON: 1,
            MessageType.CONTROL_CHANGE: 2,
            MessageType.PITCHWHEEL: 3,
        }
        self.type_lookup = {value: key for key, value in self.type_map.items()}

    def program_type_to_token(self, program: int,
                              message_type: MessageType) -> int:
        if 0 > program or program >= self.num_program:
            raise InvalidProgramError
        if message_type not in self.type_map:
            raise InvalidTypeError
        return self.type_map[
            message_type] * self.num_program + program + self.special_limit

    def note_to_token(self, note: int) -> int:
        if 0 > note or note >= self.num_note:
            raise InvalidNoteError
        return note + self.program_type_limit

    def velocity_to_token(self, velocity: int) -> int:
        if 0 > velocity or velocity >= self.num_velocity:
            raise InvalidVelocityError
        return velocity + self.note_limit

    def control_to_token(self, control: int) -> int:
        if 0 > control or control >= self.num_control:
            raise InvalidControlError
        return control + self.velocity_limit

    def value_to_token(self, value: int) -> int:
        if 0 > value or value >= self.num_value:
            raise InvalidValueError
        return value + self.control_limit

    def pitch_to_token(self, pitch: int) -> Tuple[int, int]:
        if -8192 > pitch or pitch > 8191:
            raise InvalidPitchError
        pitch_1 = (pitch >> 7) & 127
        pitch_2 = pitch & 127
        return pitch_1 + self.value_limit, pitch_2 + self.pitch_1_limit

    def tick_to_token(self, tick: int) -> int:
        if 0 > tick or tick >= self.num_tick:
            raise InvalidTickError
        return tick + self.pitch_2_limit

    def tokenize(self, event_list: List[Event]) -> ndarray:
        token_list = [self.begin]
        prev_tick = 0
        for event in event_list:
            token_list.append(
                self.program_type_to_token(event.program, event.type))
            if event.note is not None:
                token_list.append(self.note_to_token(event.note))
            if event.velocity is not None:
                token_list.append(self.velocity_to_token(event.velocity))
            if event.control is not None:
                token_list.append(self.control_to_token(event.control))
            if event.value is not None:
                token_list.append(self.value_to_token(event.value))
            if event.pitch is not None:
                token_list.extend(self.pitch_to_token(event.pitch))
            tick_delta = min(event.tick - prev_tick, self.num_tick)
            if tick_delta > 0:
                token_list.append(self.tick_to_token(tick_delta - 1))
            prev_tick = event.tick
        token_list.append(self.end)

        return np.array(token_list, dtype=np.int64)

    def tokens_to_events(self, tokens: ndarray) -> List[Event]:
        event_list: List[Event] = []
        prev_tick = 0
        pitch_1 = -1
        for token in tokens:
            if token < self.special_limit:
                continue
            if token < self.program_type_limit:
                message_type = self.type_lookup[(token - self.special_limit) //
                                                self.num_program]
                program = (token - self.special_limit) % self.num_program
                event_list.append(
                    Event(message_type=message_type,
                          tick=prev_tick,
                          program=program))
            elif token < self.note_limit:
                note = token - self.program_type_limit
                if event_list:
                    event_list[-1].note = note
            elif token < self.velocity_limit:
                velocity = token - self.note_limit
                if event_list:
                    event_list[-1].velocity = velocity
            elif token < self.control_limit:
                control = token - self.velocity_limit
                if event_list:
                    event_list[-1].control = control
            elif token < self.value_limit:
                value = token - self.control_limit
                if event_list:
                    event_list[-1].value = value
            elif token < self.pitch_1_limit:
                pitch_1 = token - self.value_limit
            elif token < self.pitch_2_limit:
                pitch_2 = token - self.pitch_1_limit
                if pitch_1 >= 0:
                    pitch = (pitch_1 << 7) + pitch_2
                    if pitch >> 13:
                        pitch |= -8192
                    if event_list:
                        event_list[-1].pitch = pitch
                    pitch_1 = -1
            elif token < self.tick_limit:
                tick = token - self.pitch_2_limit + 1
                prev_tick += tick
                if event_list:
                    event_list[-1].tick = prev_tick
            else:
                raise InvalidTokenError

        return event_list


def prepare_data(cfg: DictConfig) -> None:
    data_dir = Path(to_absolute_path(Path(*cfg.data.data_dir)))
    file_dir = Path(to_absolute_path(Path(*cfg.data.file_dir)))
    process_dir = Path(to_absolute_path(Path(*cfg.data.process_dir)))
    text_path = file_dir.joinpath("midi.txt")
    data_path = data_dir.joinpath("**", "*.mid")
    if not os.path.isdir(process_dir):
        os.makedirs(process_dir)
        filenames = []
        tokenizer = Tokenizer(cfg)
        for path in tqdm(glob.iglob(str(data_path), recursive=True)):
            relative_path = Path(path).relative_to(data_dir)
            try:
                filename = process_dir.joinpath(
                    relative_path.with_suffix(".npy"))
                os.makedirs(filename.parent, exist_ok=True)
                midi_list = read_midi(MidiFile(filename=path, clip=True))
                tokens = tokenizer.tokenize(midi_list)
                np.save(filename, tokens.astype(np.int16))
                filenames.append(str(relative_path) + "\n")
            except (EOFError, KeySignatureError, IndexError) as exception:
                tqdm.write(f"{type(exception).__name__}: {relative_path}")
                continue

        filenames.sort()
        with open(text_path, mode="w", encoding="utf-8") as file:
            file.writelines(filenames)


def read_midi(midi_file: MidiFile) -> List[Event]:
    events: List[Event] = []
    messages = []
    programs = np.zeros(16, dtype=np.int64)
    programs[9] = 128

    for track in midi_file.tracks:
        cur_tick = 0
        for message in track:
            cur_tick += message.time
            messages.append((cur_tick, message))

    messages.sort(key=itemgetter(0))

    for tick, message in messages:
        if message.type == "note_on" and message.velocity > 0:
            events.append(
                Event(MessageType.NOTE_ON,
                      tick,
                      programs[message.channel],
                      note=message.note,
                      velocity=message.velocity))
        elif message.type == "note_off" or message.type == "note_on" and message.velocity == 0:
            events.append(
                Event(MessageType.NOTE_OFF,
                      tick,
                      programs[message.channel],
                      note=message.note))
        elif message.type == "program_change" and message.channel != 9:
            programs[message.channel] = message.program
        elif message.type == "control_change":
            events.append(
                Event(MessageType.CONTROL_CHANGE,
                      tick,
                      programs[message.channel],
                      control=message.control,
                      value=message.value))
        elif message.type == "pitchwheel":
            events.append(
                Event(MessageType.PITCHWHEEL,
                      tick,
                      programs[message.channel],
                      pitch=message.pitch))

    events.sort(key=attrgetter("tick", "program", "type", "note", "velocity",
                               "control", "value", "pitch"))

    return events


def write_midi(event_list: List[Event]) -> MidiFile:
    midi_file = MidiFile()
    midi_file.add_track()
    prev_tick = 0
    program_set = set()
    for event in event_list:
        if event.program != 128:
            program_set.add(event.program)

    if len(program_set) > 15:
        print("More than 15 non-drum instruments not supported")
        return midi_file

    program_map = {}
    for i, program in enumerate(program_set):
        program_map[program] = i if i < 9 else i + 1

    for track in midi_file.tracks:
        for program, channel in program_map.items():
            track.append(
                Message("program_change",
                        channel=channel,
                        program=program,
                        time=0))

    program_map[128] = 9

    for track in midi_file.tracks:
        for event in event_list:
            channel = program_map[event.program]
            if event.type == MessageType.NOTE_OFF:
                track.append(
                    Message("note_off",
                            channel=channel,
                            note=event.note,
                            velocity=event.velocity,
                            time=event.tick - prev_tick))
            elif event.type == MessageType.NOTE_ON:
                track.append(
                    Message("note_on",
                            channel=channel,
                            note=event.note,
                            velocity=event.velocity,
                            time=event.tick - prev_tick))
            elif event.type == MessageType.CONTROL_CHANGE:
                track.append(
                    Message("control_change",
                            channel=channel,
                            control=event.control,
                            value=event.value,
                            time=event.tick - prev_tick))
            elif event.type == MessageType.PITCHWHEEL:
                track.append(
                    Message("pitchwheel",
                            channel=channel,
                            pitch=event.pitch,
                            time=event.tick - prev_tick))
            else:
                raise InvalidTypeError
            prev_tick = event.tick

    return midi_file
