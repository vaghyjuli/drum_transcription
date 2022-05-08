import glob
import os
import sys
import mido
from mido import MidiFile
from mido import Message
import matplotlib.pyplot as plt


class Sample:
    def __init__(self, _dir, _bpm, _midi_file, _wav_file, _instrument_codes):
        self.dir = _dir
        self.bpm = _bpm
        self.midi_file = _midi_file
        self.wav_file = _wav_file
        self.instrument_codes = _instrument_codes
        self.init_midi_labels()

    def __str__(self):
        return self.dir

    def __repr__(self):
        return self.dir

    def init_midi_labels(self):
        mid = MidiFile(self.midi_file, clip=True)
        drums = set()
        for msg in mid.tracks[0]:
            if not msg.is_meta:
                drums.add(msg.note)
        self.labels = MIDILabels(list(drums))
        ticks = 0
        time_signature_msg = 0
        tempo = 0
        for msg in mid.tracks[0]:
            if msg.is_meta:
                if msg.type == 'time_signature':
                    time_signature_msg = msg
                elif msg.type == 'set_tempo':
                    tempo = msg.tempo
                continue
            ticks += msg.time
            if msg.type == 'note_on':
                self.labels.add_onset(msg.note, ticks)
        #self.labels.print_onsets()
        # DO: the conversion below might be incorrect
        #self.labels.plot((mid.length/ticks) * (120/self.bpm))


class MIDILabels:
    def __init__(self, _instrument_labels):
        self.instruments = {}
        colors = ["blue", "green", "cyan", "magenta", "yellow", "black"]
        for idx in range(len(_instrument_labels)):
            self.instruments[_instrument_labels[idx]] = self.Instrument(_instrument_labels[idx], colors[idx%len(colors)], idx)

    def add_onset(self, instrument_label, onset):
        self.instruments[instrument_label].add_onset(onset)

    def print_onsets(self):
        for label, instrument in self.instruments.items():
            instrument.print_onsets()

    def plot(self, tick_duration):
        fig, ax = plt.subplots(1)
        for label, instrument in self.instruments.items():
            instrument.plot(tick_duration)
        ax.set_yticklabels([])
        plt.xlabel('Time (s)')
        plt.show()

    class Instrument:
        def __init__(self, _label, _color, _idx):
            self.label = _label
            self.color = _color
            self.idx = _idx
            self.onsets = []

        def __str__(self):
            return self.label

        def print_onsets(self):
            print(f"{self.label}: {self.onsets}")

        def add_onset(self, onset):
            self.onsets.append(onset)

        def plot(self, tick_duration):
            for onset in self.onsets:
                plt.plot(onset*tick_duration, (self.idx+1)*(200), 'o', color=self.color)
                # convert to seconds


class NMFLabels:
    def __init__(self):
        pass


def read_data(data_folder):
    samples = []
    os.chdir(data_folder)
    sample_directories = [item for item in os.listdir() if os.path.isdir(item)]
    for sample_directory in sample_directories:
        os.chdir(os.path.join(data_folder, sample_directory))
        make_path = lambda f : os.path.join(data_folder, sample_directory, f)
        if len(glob.glob("*.mid")) != 1:
            print(f"There should be a single MIDI file in {sample_directory}")
            continue
        midi_file = make_path(glob.glob("*.mid")[0])
        if len(glob.glob("*.wav")) != 1:
            print(f"There should be a single WAV file in {sample_directory}")
            continue
        wav_file = make_path(glob.glob("*.wav")[0])
        instrument_codes = {}
        with open ("info.txt", "r") as info_file:
            data = info_file.read().splitlines()
            n_instruments = len(data) - 4
            try:
                bpm = int(data[0].split()[0])
            except Exception as error:
                print(f"Failed to identify BPM in {sample_directory}")
                continue
            try:
                for i in range(4, len(data)):
                    instrument_codes[data[i].split()[0]] = data[i].split()[1]
            except Exception as error:
                print(f"Failed to identify instrument codes in {sample_directory}")
                continue
        os.chdir(make_path("instruments"))
        missing_instruments = [i for i in instrument_codes.values() if i not in glob.glob("*.wav")]
        if len(missing_instruments) > 0:
            print(f"{missing_instruments} missing in {sample_directory}/instruments")
            continue
        for midi_code, instrument_wav in instrument_codes.items():
            instrument_codes[midi_code] = make_path(instrument_wav)
        samples.append(Sample(sample_directory, bpm, midi_file, wav_file, instrument_codes))
    return samples


data_folder = r'/Users/juliavaghy/Desktop/0-syth_data/data'
samples = read_data(data_folder)
print(f"Included folders: {samples}")