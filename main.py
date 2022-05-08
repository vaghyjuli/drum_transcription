import glob
import os
import sys
import mido
from mido import MidiFile
from mido import Message
import matplotlib.pyplot as plt


class Sample:
    """
    A class representing a single test sample.

    Args:
        _dir (str) : The name of the sample's directory.
        _bpm (int): BPM of the recording.
        _midi_file (str) : The path to the MIDI file.
        _wav_file (str) : The path to the WAV recording.
        _instrument_codes (dict of int: str) :
            key : MIDI note of instrument in the MIDI file
            value : The path to the WAV template of the instrument
            For each instrument present in the sample.

    Attributes:
        dir (str) : The name of the sample's directory.
        midi_file (str) : The path to the MIDI file.
        wav_file (str) : The path to the WAV recording.
        instrument_codes (dict of int: str) :
            key : MIDI note of instrument in the MIDI file.
            value : The path to the WAV template of the instrument.
            For each instrument present in the sample.
        midi_labels (MIDILabels) : The labels extracted from the MIDI file.

    Methods:
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
            key : MIDI note of instrument in the MIDI file.
            value : The path to the WAV template of the instrument.

    Attributes:
        bpm (int) : BPM of the MIDI file.
        instruments (dict of int: Instrument) :
            key : MIDI note of instrument.
            value : Instrument object.
        tick_duration (float): Duration of a tick.

    Raises:
        Exception : MIDI codes don't match those provided in info.txt.

    Methods:
        print_onsets()
        plot()
    """

    def __init__(self, _midi_file, _bpm, _instrument_codes):
        self.bpm = _bpm
        mid = MidiFile(_midi_file, clip=True)
        midi_notes = set()
        for msg in mid.tracks[0]:
            if not msg.is_meta:
                midi_notes.add(msg.note) # each drum has its corresponding note in the MIDI file
        midi_notes = sorted(list(midi_notes))
        if midi_notes != sorted(list(_instrument_codes.keys())):
            raise Exception("MIDI codes don't match those provided in info.txt")
        self.instruments = {}
        colors = ["blue", "green", "cyan", "magenta", "yellow", "black"]
        for idx in range(len(midi_notes)):
            self.instruments[midi_notes[idx]] = Instrument(midi_notes[idx], colors[idx%len(colors)], idx)
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
                self.instruments[msg.note].add_onset(ticks)
        # DO: the conversion below might be incorrect
        self.tick_duration = (mid.length/ticks) * (120/self.bpm)

    def print_onsets(self):
        """
        Prints the onset ticks for each instrument.
            Instrument MIDI note: [onset ticks]
        """
        for midi_note, instrument in self.instruments.items():
            instrument.print_onsets()

    def plot(self):
        """
        Visualizes the MIDI file's onsets in a plot.
        """
        fig, ax = plt.subplots(1)
        for midi_note, instrument in self.instruments.items():
            instrument.plot(self.tick_duration)
        ax.set_yticklabels([])
        plt.xlabel('Time (s)')
        plt.show()


class Instrument:
    """
    A class storing the onsets and MIDI note of a given instrument in a sample.

    Args:
        _midi_note (int): MIDI note of instrument.
        _color (str): Allocated color, to be used for plotting.
        _idx (int) : Allocated instrument index in the sample, to be used for plotting.

    Attributes:
        midi_note (int): MIDI note of instrument.
        color (str): Allocated color, to be used for plotting.
        idx (int) : Allocated instrument index in the sample, to be used for plotting.
        onsets (ndarray int): Array containig the onset ticks of the instrument.

    Methods:
        print_onsets()
        add_onset()
        plot()
        compare()
    """
    def __init__(self, _midi_note, _color, _idx):
        self.midi_note = _midi_note
        self.color = _color
        self.idx = _idx
        self.onsets = [] # tick

    def __str__(self):
        return self.midi_note

    def print_onsets(self):
        """
        Print the instrument's MIDI onset ticks.
            Instrument MIDI note: [onset ticks]
        """
        print(f"{self.midi_note}: {self.onsets}")

    def add_onset(self, onset):
        self.onsets.append(onset)

    def plot(self, tick_duration):
        for onset in self.onsets:
            plt.plot(onset*tick_duration, (self.idx+1)*(200), 'o', color=self.color)
            # convert to seconds

    def compare(self, other):
        """
        Evaluate distance to another Instrument object
        """
        for onset in other.onsets:
            pass


class NMFLabels:
    def __init__(self):
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
                for i in range(4, len(data)):
                    instrument_codes[int(data[i].split()[0])] = data[i].split()[1]
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