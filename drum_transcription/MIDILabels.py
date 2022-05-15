import matplotlib.pyplot as plt
import mido
from mido import MidiFile
from mido import Message

from Instrument import Instrument

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
        # TODO: the conversion below might be incorrect
        #print(self.bpm, mid.length * (120/self.bpm))
        self.tick_duration = (mid.length/ticks) * (120/self.bpm)
        for midi_note, instrument in self.instrument_codes.items():
            instrument.set_tick_duration(self.tick_duration)

    def print_onsets(self):
        """
            Prints the onset ticks for each instrument.
                Instrument MIDI note: [onset ticks]
        """
        for midi_note, instrument in self.instrument_codes.items():
            instrument.print_onsets()

    def plot(self):
        """
            Visualizes the MIDI file's onsets in a plot.
        """
        fig, ax = plt.subplots(1)
        for midi_note, instrument in self.instrument_codes.items():
            instrument.plot_midi()
        ax.set_yticklabels([])
        plt.xlabel('Time (s)')
        plt.show()