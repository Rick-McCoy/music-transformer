from enum import IntEnum
import glob
from operator import attrgetter, itemgetter
import os
import pathlib
import random
from typing import List, Optional

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


class InvalidPitchError(Exception):
    pass


class InvalidVelocityError(Exception):
    pass


class InvalidTickError(Exception):
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


class Note:
    def __init__(self, message_type, tick, pitch, velocity, program):
        self.type: MessageType = message_type
        self.tick: int = tick
        self.pitch: int = pitch
        self.velocity: int = velocity
        self.program: int = program


class Tokenizer:
    def __init__(self, cfg: DictConfig) -> None:
        self.pad = 0
        self.begin = 1
        self.end = 2
        self.num_program = cfg.model.num_program
        self.num_type = cfg.model.num_type
        self.num_program_type = self.num_program * self.num_type
        self.num_pitch = cfg.model.num_pitch
        self.num_velocity = cfg.model.num_velocity
        self.num_tick = cfg.model.num_tick
        self.valid_types = [MessageType.NOTE_OFF, MessageType.NOTE_ON]

    def program_type_to_token(self, program: int,
                              message_type: MessageType) -> int:
        lower_bound = 3
        if 0 > program or program >= self.num_program:
            raise InvalidProgramError
        if message_type not in self.valid_types:
            raise InvalidTypeError
        return (message_type - 1) * self.num_program + program + lower_bound

    def pitch_to_token(self, pitch: int) -> int:
        lower_bound = self.num_program_type + 3
        if 0 > pitch or pitch >= self.num_pitch:
            raise InvalidPitchError
        return pitch + lower_bound

    def velocity_to_token(self, velocity: int) -> int:
        lower_bound = self.num_pitch + self.num_program_type + 3
        if 0 > velocity or velocity >= self.num_velocity:
            raise InvalidVelocityError
        return velocity + lower_bound

    def tick_to_token(self, tick: int) -> int:
        lower_bound = self.num_velocity + self.num_pitch + self.num_program_type + 3
        if 0 > tick or tick >= self.num_tick:
            raise InvalidTickError
        return tick + lower_bound

    def tokenize(self,
                 note_list: List[Note],
                 length: Optional[int] = None) -> ndarray:
        token_list = [self.begin]
        prev_tick = 0
        for note in note_list:
            # token_list.append(self.program_to_token(note.program))
            # token_list.append(self.type_to_token(note.type))
            token_list.append(
                self.program_type_to_token(note.program, note.type))
            token_list.append(self.pitch_to_token(note.pitch))
            if note.type == MessageType.NOTE_ON:
                token_list.append(self.velocity_to_token(note.velocity))
            tick_delta = min(note.tick - prev_tick, self.num_tick)
            if tick_delta > 0:
                token_list.append(self.tick_to_token(tick_delta - 1))
            prev_tick = note.tick
        token_list.append(self.end)

        if length is not None:
            orig_len = len(token_list)
            if length > orig_len:
                return np.pad(token_list, (0, length - orig_len),
                              mode="constant",
                              constant_values=self.pad)
            random_index = random.randint(0, orig_len - length)
            return np.array(token_list[random_index:random_index + length],
                            dtype=np.int64)

        return np.array(token_list, dtype=np.int64)

    def tokens_to_notes(self, tokens: ndarray) -> List[Note]:
        note_list: List[Note] = []
        prev_tick = 0
        for token in tokens:
            if token < 3:
                continue
            if token < self.num_program_type + 3:
                message_type = MessageType((token - 3) // self.num_program + 1)
                program = (token - 3) % self.num_program
                note_list.append(
                    Note(message_type=message_type,
                         tick=prev_tick,
                         pitch=0,
                         velocity=0,
                         program=program))
            elif token < self.num_pitch + self.num_program_type + 3:
                pitch = token - self.num_program_type - 3
                if note_list:
                    note_list[-1].pitch = pitch
            elif token < self.num_velocity + self.num_pitch + self.num_program_type + 3:
                velocity = token - self.num_pitch - self.num_program_type - 3
                if note_list:
                    note_list[-1].velocity = velocity
            elif token < self.num_tick + self.num_velocity + self.num_pitch + self.num_program_type + 3:
                tick = token - self.num_velocity - self.num_pitch - self.num_program_type - 3 + 1
                prev_tick += tick
                if note_list:
                    note_list[-1].tick = prev_tick

        return note_list


def prepare_data(cfg: DictConfig) -> None:
    data_dir = os.path.join(*cfg.data.data_dir)
    file_dir = os.path.join(*cfg.data.file_dir)
    os.makedirs(file_dir, exist_ok=True)
    file_path = to_absolute_path(os.path.join(file_dir, "midi.npz"))
    text_path = to_absolute_path(os.path.join(file_dir, "midi.txt"))
    data_path = to_absolute_path(os.path.join(data_dir, "**", "*.mid"))
    if not os.path.isfile(file_path):
        with open(text_path, mode="w", encoding="utf-8") as file:
            tokens = []
            tokenizer = Tokenizer(cfg)
            count = 0
            for path in tqdm(glob.iglob(data_path, recursive=True)):
                relative_path = pathlib.Path(path).relative_to(
                    to_absolute_path(data_dir))
                try:
                    tokens.append(
                        tokenizer.tokenize(
                            read_midi(MidiFile(filename=path,
                                               clip=True))).astype(np.int16))
                    file.write(str(relative_path) + "\n")
                    count += 1
                except (EOFError, KeySignatureError, IndexError) as exception:
                    tqdm.write(f"{type(exception).__name__}: {relative_path}")
                    continue
                if count > 100:
                    break

        np.savez(to_absolute_path(file_path), *tokens)


def read_midi(midi_file: MidiFile) -> List[Note]:
    notes: List[Note] = []
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
            program = programs[message.channel]
            notes.append(
                Note(MessageType.NOTE_ON, tick, message.note, message.velocity,
                     program))
        elif message.type == "note_off" or message.type == "note_on" and message.velocity == 0:
            program = programs[message.channel]
            notes.append(
                Note(MessageType.NOTE_OFF, tick, message.note, 0, program))
        elif message.type == "program_change" and message.channel != 9:
            programs[message.channel] = message.program

    notes.sort(key=attrgetter("tick", "program", "type", "pitch", "velocity"))

    return notes


def write_midi(note_list: List[Note]) -> MidiFile:
    midi_file = MidiFile()
    midi_file.add_track()
    prev_tick = 0
    programs = np.full(16, fill_value=-1, dtype=np.int64)
    for track in midi_file.tracks:
        for note in note_list:
            channel = 9 if note.program == 128 else 0
            if note.program != programs[channel] and channel != 9:
                track.append(
                    Message("program_change",
                            channel=channel,
                            program=note.program,
                            time=note.tick - prev_tick))
                programs[channel] = note.program
                prev_tick = note.tick
            if note.type == MessageType.NOTE_OFF:
                track.append(
                    Message("note_off",
                            channel=channel,
                            note=note.pitch,
                            velocity=note.velocity,
                            time=note.tick - prev_tick))
            elif note.type == MessageType.NOTE_ON:
                track.append(
                    Message("note_on",
                            channel=channel,
                            note=note.pitch,
                            velocity=note.velocity,
                            time=note.tick - prev_tick))
            else:
                raise InvalidTypeError
            prev_tick = note.tick

    return midi_file
