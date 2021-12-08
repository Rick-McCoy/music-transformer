import bisect
from enum import IntEnum
import glob
from operator import attrgetter
import os
import pathlib
import random
from time import time
from typing import List, Tuple

from hydra import initialize, compose
from hydra.utils import to_absolute_path
import mido
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
    def __init__(self, message_type, tick, pitch, velocity, channel, program):
        self.type: MessageType = message_type
        self.tick: int = tick
        self.pitch: int = pitch
        self.velocity: int = velocity
        self.channel: int = channel
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
                    mido.MidiFile(path, clip=True)
                    file.write(relative_path + "\n")
                except (EOFError, KeySignatureError, IndexError) as exception:
                    tqdm.write(f"{type(exception).__name__}: {relative_path}")
                    continue


def read_midi(
    midi_file: mido.MidiFile
) -> Tuple[ndarray, ndarray, ndarray, ndarray, ndarray]:
    notes: List[Note] = []
    channel_program_log = [[(0, 0)] for _ in range(17)]
    for track in midi_file.tracks:
        track: mido.MidiTrack
        cur_tick = 0
        for message in track:
            message: mido.Message
            cur_tick += message.time
            if message.type == "note_on" and message.velocity > 0:
                notes.append(
                    Note(MessageType.NOTE_ON, cur_tick, message.note,
                         message.velocity, message.channel, -1))
            elif message.type == "note_off" or message.type == "note_on" and message.velocity == 0:
                notes.append(
                    Note(MessageType.NOTE_OFF, cur_tick, message.note, 0,
                         message.channel, -1))
            elif message.type == "program_change":
                channel_program_log[message.channel].append(
                    (cur_tick, message.program))

    _ = [i.sort() for i in channel_program_log]
    channel_program_history = [
        list(map(list, zip(*i))) for i in channel_program_log
    ]

    for history in channel_program_history:
        history[0] = history[0][1:]

    for note in notes:
        tick_list = channel_program_history[note.channel][0]
        program_list = channel_program_history[note.channel][1]
        note.program = program_list[bisect.bisect(tick_list, note.tick)]

    notes.sort(key=attrgetter("tick", "program", "type", "pitch", "velocity"))

    ticks = np.diff([0] + [note.tick for note in notes])
    programs = np.array([note.program for note in notes])
    types = np.array([note.type for note in notes])
    pitches = np.array([note.pitch for note in notes])
    velocities = np.array([note.velocity for note in notes])

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
            ticks.extend(
                read_midi(
                    mido.MidiFile(filename=to_absolute_path(path.strip()),
                                  clip=True))[0])
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
