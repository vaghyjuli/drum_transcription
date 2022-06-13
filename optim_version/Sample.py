from MIDILabels import MIDILabels
from NMFLabels import NMFLabels
from Instrument import Instrument

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
    def __init__(self, _params, _dir, _bpm, _midi_file, _wav_file, _instrument_codes):
        self.dir = _dir
        self.instrument_codes = _instrument_codes
        self.midi_labels = MIDILabels(_midi_file, _bpm, _instrument_codes)
        self.nmf_labels = NMFLabels(_params, _wav_file, _instrument_codes)
        #self.midi_labels.print_onsets()
        #self.midi_labels.plot()
        #self.nmf_labels.plot_instrument_spectra()
        #self.nmf_labels.plot_template_matrix()
        #self.nmf_labels.plot_recording_spectrum()

    def __str__(self):
        return self.dir

    def __repr__(self):
        return self.dir

    def evaluate(self, comment=False):
        tp_count = 0
        fp_count = 0
        fn_count = 0
        for midi_note, instrument in self.instrument_codes.items():
            instrument.evaluate()
            tp_count += instrument.tp_count
            fp_count += instrument.fp_count
            fn_count += instrument.fn_count
        f_measure = (2*tp_count) / (2*tp_count + fp_count + fn_count)
        precision = tp_count / (tp_count + fp_count)
        recall = tp_count / (tp_count + fn_count)
        if comment:
            print(f"TP={tp_count}, FP={fp_count}, FN={fn_count}")
            print(f"precision = {precision}")
            print(f"recall = {recall}")
            print(f"F-measure = {f_measure}")
        return precision, recall, f_measure