import glob
import os
import sys
import mido
from mido import MidiFile
from mido import Message
import matplotlib.pyplot as plt
import librosa
import numpy as np


class Sample:
    """
        A class representing a single test sample.

        Args:
            _dir (str) : The name of the sample's directory.
            _bpm (int): BPM of the recording.
            _midi_file (str) : The path to the MIDI file.
            _wav_file (str) : The path to the WAV recording.
            _instrument_codes (dict of int: Instrument) :
                key : the note of the instrument in the MIDI file.
                value : The instrument object.

        Attributes:
            dir (str) : The name of the sample's directory.
            midi_file (str) : The path to the MIDI file.
            wav_file (str) : The path to the WAV recording.
            instrument_codes (dict of int: str) :
                key : The note of the instrument in the MIDI file.
                value : The instrument object.
            midi_labels (Labels) : The labels extracted from the MIDI file.
    """
    def __init__(self, _dir, _bpm, _midi_file, _wav_file, _instrument_codes):
        self.dir = _dir
        self.midi_file = _midi_file
        self.wav_file = _wav_file
        self.instrument_codes = _instrument_codes
        self.midi_labels = MIDILabels(_midi_file, _bpm, _instrument_codes)
        #self.midi_labels.print_onsets()
        #self.midi_labels.plot()

    def __str__(self):
        return self.dir

    def __repr__(self):
        return self.dir


class MIDILabels:
    """
        A class storing the MIDI labels of a sample.

        Args:
            _midi_file (str): The path to the MIDI file.
            _bpm (int): BPM of the recording.
            _instrument_codes (dict of int: str) :
                key : The note of instrument in the MIDI file.
                value : The path to the WAV template of the instrument.

        Attributes:
            bpm (int) : BPM of the MIDI file.
            instrument_codes (dict of int: Instrument) :
                key : The note of instrument in the MIDI file.
                value : The instrument object.
            tick_duration (float): Duration of a tick.

        Raises:
            Exception : MIDI codes don't match those provided in info.txt.

        Methods:
            print_onsets()
            plot()
    """

    def __init__(self, _midi_file, _bpm, _instrument_codes):
        self.bpm = _bpm
        self.instrument_codes = _instrument_codes
        mid = MidiFile(_midi_file, clip=True)
        midi_notes = set()
        for msg in mid.tracks[0]:
            if not msg.is_meta:
                midi_notes.add(msg.note) # each drum has its corresponding note in the MIDI file
        midi_notes = sorted(list(midi_notes))
        if midi_notes != sorted(list(_instrument_codes.keys())):
            raise Exception("MIDI codes don't match those provided in info.txt")
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
                self.instrument_codes[msg.note].add_midi_onset(ticks)
        # DO: the conversion below might be incorrect
        self.tick_duration = (mid.length/ticks) * (120/self.bpm)

    # rewrite to print_midi_onsets()
    def print_onsets(self):
        """
            Prints the onset ticks for each instrument.
                Instrument MIDI note: [onset ticks]
        """
        for midi_note, instrument in self.instrument_codes.items():
            instrument.print_onsets()

    # rewrite to plot_midi()
    def plot(self):
        """
            Visualizes the MIDI file's onsets in a plot.
        """
        fig, ax = plt.subplots(1)
        for midi_note, instrument in self.instrument_codes.items():
            instrument.plot(self.tick_duration)
        ax.set_yticklabels([])
        plt.xlabel('Time (s)')
        plt.show()


class NMFLabels:
    def __init__(self, _wav_file, _instrument_codes):
        self.wav_file = wav_file
        pass


class Instrument:
    """
        A class storing the onsets and MIDI note of a given instrument in a sample.

        Args:
            _midi_note (int): MIDI note of instrument.
            _color (str): Allocated color, to be used for plotting.
            _idx (int) : Allocated instrument index in the sample, to be used for plotting.
            _wav_file (str) : Path to the instrument's template recording.

        Attributes:
            midi_note (int): MIDI note of instrument.
            color (str): Allocated color, to be used for plotting.
            idx (int) : Allocated instrument index in the sample, to be used for plotting.
            onsets (ndarray int): Array containig the onset ticks of the instrument.
            wav_file (str) : Path to the instrument's template recording.
            N (int) : STFT window size
            H (int) : STFT hop size

        Methods:
            print_onsets()
            add_midi_onset()
            plot()
            compare()
    """
    def __init__(self, _midi_note, _color, _idx, _wav_file):
        self.midi_note = _midi_note
        self.color = _color
        self.idx = _idx
        self.midi_onsets = [] # tick
        self.wav_file = _wav_file
        self.N = 512
        # TODO: fix N! - change in NMF algo
        self.H = 256
        self.init_template()

    def __str__(self):
        return self.midi_note

    def print_onsets(self):
        """
            Print the instrument's MIDI onset ticks.
                Instrument MIDI note: [onset ticks]
        """
        print(f"{self.midi_note}: {self.midi_onsets}")

    def add_midi_onset(self, onset):
        self.midi_onsets.append(onset)

    def init_template(self):
        x, self.Fs = librosa.load(self.wav_file)
        X = librosa.stft(x, n_fft=self.N, hop_length=self.H, win_length=self.N, window='hann', center=True, pad_mode='constant')
        self.Y = np.log(1 + 10 * np.abs(X)) # self.Y.shape = (1025, T)
        #self.Y = self.Y[0:100,:] # self.Y.shape = (100, T)
        ## TODO: OPTIMIZE normalization
        #self.Y = normalize_feature_sequence(X=self.Y, norm="z")
        #print(self.Y.shape)
        self.template = np.mean(self.Y, axis=1)
        #print(self.template)

    def plot(self, tick_duration):
        for onset in self.midi_onsets:
            plt.plot(onset*tick_duration, (self.idx+1)*(200), 'o', color=self.color)
            # convert to seconds

    def compare(self, other):
        """
        Evaluate distance to another Instrument object
        """
        for onset in other.onsets:
            pass


def read_data(data_folder):
    """
        Main loop for reading data. If data in a sample does not align with the required format, it is skipped
        (not included in evaluation), and the user is notified via a message printed to the terminal.

        Parameters:
            data_folder (srt): The main folder in which the samples are located, structured as follows.

                data_folder
                |   
                +-- 1
                |  |  
                |  +-- sample.mid
                |  +-- sample.wav
                |  +-- info.txt
                |  \-- instruments
                |       |
                |       +-- instrument1.wav
                |       +-- instrument2.wav
                |       +-- instrumen3.wav
                |       +-- ...
                +-- 2
                |   |
                |  +-- sample.mid
                |  +-- sample.wav
                |  +-- info.txt
                |  \-- instruments
                |       |
                |       +-- instrument1.wav
                |       +-- instrument2.wav
                |       +-- instrumen3.wav
                |       +-- ...
                |
                +-- 3
                |   |
                |   +-- sample.mid
                | ...

        Returns:
            ndarray Sample : An array of Sample objects, extracted from the specified data folder. 
    """
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
                # BPM is not stored in MIDI files exported from Ableton
                # so it has to be extracted from info.txt
            except Exception as error:
                print(f"Failed to identify BPM in {sample_directory}")
                continue
            try:
                colors = ["blue", "green", "cyan", "magenta", "yellow", "black"]
                info_instruments = []
                for i in range(4, len(data)):
                    midi_note = int(data[i].split()[0])
                    info_instruments.append(data[i].split()[1])
                    instrument_wav = make_path(f"instruments/{data[i].split()[1]}")
                    instrument_codes[midi_note] = Instrument(midi_note, colors[i-4], i-4, instrument_wav)
            except Exception as error:
                print(f"Failed to identify instrument codes in {sample_directory}")
                continue
        os.chdir(make_path("instruments"))
        missing_instruments = [instrument_wav for instrument_wav in info_instruments if instrument_wav not in glob.glob("*.wav")]
        if len(missing_instruments) > 0:
            print(f"{missing_instruments} missing in {sample_directory}/instruments")
            continue
        try:
            samples.append(Sample(sample_directory, bpm, midi_file, wav_file, instrument_codes))
        except Exception as error:
            print(f"{error} in {sample_directory}")
            continue
    return samples


data_folder = r'/Users/juliavaghy/Desktop/0-syth_data/data'
samples = read_data(data_folder)
print(f"Included folders: {samples}")
#print(Sample.__doc__)