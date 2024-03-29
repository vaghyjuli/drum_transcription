from mido import MidiFile
from mido import Message
import mido
import matplotlib.pyplot as plt

f_name = input("path = ")
#f_name = "/Users/juliavaghy/Desktop/0--synth/data/1/1.mid"
#f_name = "/Users/juliavaghy/Desktop/OddGrooves/Oddgrooves Free Grooves for Toontrack/02 Reggae Drumming/065 Chorus 010 Outro (17).mid"
mid = MidiFile(f_name, clip=True)
print(f"{len(mid.tracks)} track(s)")

class Labels:
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

        def add_onset(self, onset):
            self.onsets.append(onset)

        def print_onsets(self):
            #print(self.label, self.color, self.onsets)
            print(self.label, self.color)

        def plot(self, tick_duration):
            for onset in self.onsets:
                plt.plot(onset*tick_duration, (self.idx+1)*(200), 'o', color=self.color)
                # convert to seconds

drums = set()

for msg in mid.tracks[0]:
    #print(msg)
    if msg.type == 'note_on':
        drums.add(msg.note)


print(f"drum tracks = {drums}")
labels = Labels(list(drums))

ticks = 0
time_signature_msg = 0
tempo = 0
for msg in mid.tracks[0]:
    if msg.type != "note_on" and msg.type != "note_off":
        print(msg)
    if msg.is_meta:
        #print(msg)
        if msg.type == 'time_signature':
            time_signature_msg = msg
        elif msg.type == 'set_tempo':
            tempo = msg.tempo
        continue
    ticks += msg.time
    if msg.type == 'note_on':
        labels.add_onset(msg.note, ticks)

labels.print_onsets()
labels.plot(mid.length/ticks)

"""
if tempo == 0: tempo = mido.bpm2tempo(120)
print(f"time signature {time_signature_msg.numerator}/{time_signature_msg.denominator}")
print(f"clocks_per_click = {time_signature_msg.clocks_per_click}")
print(f"notated_32nd_notes_per_beat = {time_signature_msg.notated_32nd_notes_per_beat}")
print(f"tempo = {tempo} microsecond/beat = {tempo/(10**6)} s/beat")
print(f"{mid.length} seconds")
print(f"{mid.length/(tempo/(10**6))} {mido.tempo2bpm(tempo)} BPM")
print(f"{ticks} ticks")

print(f"{mid.ticks_per_beat} ticks per beat")
print(f"tick duration = {mid.length/ticks} s")
"""