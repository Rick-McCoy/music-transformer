from enum import IntEnum
import glob
from operator import attrgetter, itemgetter
import os
import pathlib
import random
from time import time
from typing import List, Tuple

from hydra import initialize, compose
from hydra.utils import to_absolute_path
from mido import MidiFile
from mido.midifiles.meta import KeySignatureError
import numpy as np
from numpy import ndarray
from tqdm import tqdm
# import matplotlib.pyplot as plt


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


def prepare_data(data_dir: str, file_dir: str) -> None:
    os.makedirs(file_dir, exist_ok=True)
    file_path = to_absolute_path(os.path.join(file_dir, "midi.txt"))
    data_path = to_absolute_path(os.path.join(data_dir, "**", "*.mid"))
    if not os.path.isfile(file_path):
        with open(file_path, mode="w", encoding="utf-8") as file:
            for path in tqdm(glob.iglob(data_path, recursive=True)):
                relative_path = pathlib.Path(path).relative_to(data_dir)
                try:
                    MidiFile(path, clip=True)
                    file.write(relative_path + "\n")
                except (EOFError, KeySignatureError, IndexError) as exception:
                    tqdm.write(f"{type(exception).__name__}: {relative_path}")
                    continue


def read_midi(
        midi_file: MidiFile
) -> Tuple[ndarray, ndarray, ndarray, ndarray, ndarray]:
    notes: List[Note] = []
    messages = []
    programs = np.zeros(16, dtype=np.int64)

    for track in midi_file.tracks:
        cur_tick = 0
        for message in track:
            cur_tick += message.time
            if hasattr(message, "channel") and message.channel == 9:
                continue
            messages.append((cur_tick, message))

    messages.sort(key=itemgetter(0))
    note_velocity = np.zeros((128, 128), dtype=np.int64)

    for tick, message in messages:
        if message.type == "note_on" and message.velocity > 0:
            program = programs[message.channel]
            notes.append(
                Note(MessageType.NOTE_ON, tick, message.note, message.velocity,
                     program))
            note_velocity[program, message.note] = message.velocity
        elif message.type == "note_off" or message.type == "note_on" and message.velocity == 0:
            program = programs[message.channel]
            notes.append(
                Note(MessageType.NOTE_ON, tick, message.note,
                     note_velocity[program, message.note], program))
        elif message.type == "program_change":
            programs[message.channel] = message.program

    notes.sort(key=attrgetter("tick", "program", "type", "pitch", "velocity"))

    ticks = np.diff(
        np.array([0] + [note.tick for note in notes], dtype=np.int64))
    programs = np.array([note.program for note in notes], dtype=np.int64)
    types = np.array([note.type for note in notes], dtype=np.int64)
    pitches = np.array([note.pitch for note in notes], dtype=np.int64)
    velocities = np.array([note.velocity for note in notes], dtype=np.int64)

    return ticks, programs, types, pitches, velocities


def main():
    with initialize(config_path="../config"):
        start = time()
        ticks = []
        cfg = compose(config_name="config")
        data_dir = os.path.join(*cfg.data.data_dir)
        file_dir = os.path.join(*cfg.data.file_dir)
        prepare_data(data_dir, file_dir)
        file_path = to_absolute_path(os.path.join(file_dir, "midi.txt"))
        with open(file_path, mode="r", encoding="utf-8") as file:
            path_list = file.readlines()
        random.shuffle(path_list)
        for path in tqdm(path_list):
            filename = to_absolute_path(os.path.join(data_dir, path.strip()))
            ticks.extend(read_midi(MidiFile(filename=filename, clip=True))[0])
            if time() - start > 100:
                # plt.hist(ticks,
                #          bins=np.exp(np.linspace(np.log(1e3), np.log(1e5),
                #                                  100)))
                # plt.hist(ticks, bins=100)
                # plt.xlabel("Tick")
                # plt.ylabel("Frequency")
                # plt.xscale("log")
                # plt.savefig("temp.png")
                # plt.close()
                print(f"50%: {np.percentile(ticks, 50)}")
                print(f"75%: {np.percentile(ticks, 75)}")
                print(f"90%: {np.percentile(ticks, 90)}")
                print(f"95%: {np.percentile(ticks, 95)}")
                print(f"99%: {np.percentile(ticks, 99)}")
                print(f"99.5%: {np.percentile(ticks, 99.5)}")
                print(f"99.9%: {np.percentile(ticks, 99.9)}")
                break


if __name__ == '__main__':
    main()
