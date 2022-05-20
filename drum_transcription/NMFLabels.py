import matplotlib.pyplot as plt
import librosa
import numpy as np

from Instrument import Instrument
from nmfd import *
from nmf import *

class NMFLabels:
    def __init__(self, _params, _wav_file, _instrument_codes):
        self.wav_file = _wav_file
        self.instrument_codes = _instrument_codes
        self.window = _params["window"]
        self.hop = _params["hop"]
        self.nmf_type = _params["nmf_type"]
        self.fixW = _params["fixW"]
        self.initH = _params["initH"]
        self.calculate_STFT()
        self.initialize_template_matrix()
        self.factorize()
        #self.plot_template_matrix()
        #self.plot_activation_matrix()

    def initialize_template_matrix(self):
        if self.nmf_type == 'NMF':
            templates = [instrument.template for midi_note, instrument in self.instrument_codes.items()]
            self.W_init = np.array(templates, dtype=np.float64).transpose()
        elif self.nmf_type == 'NMFD':
            T = max([instrument.Y.shape[1] for midi_note, instrument in self.instrument_codes.items()])
            templates = [instrument.template_2D(T) for midi_note, instrument in self.instrument_codes.items()]
            self.W_init = np.array(templates, dtype=np.float64).transpose((1, 0, 2))

    def factorize(self):
        if self.nmf_type == 'NMF':
            V_approx, W, H = NMF(V=self.V, W_init=self.W_init, fixW=self.fixW)
        elif self.nmf_type == 'NMFD':
            V_approx, W, H = NMFD(V=self.V, W_init=self.W_init, fixW=self.fixW)
        i = 0
        for midi_note, instrument in self.instrument_codes.items():
            instrument.set_activation(H[i])
            instrument.find_onsets()
            i+=1

    def calculate_STFT(self):
        x, self.Fs = librosa.load(self.wav_file)
        X = librosa.stft(x, n_fft=self.window, hop_length=self.hop, win_length=self.window, window='hann', center=True, pad_mode='constant')
        self.V = np.log(1 + 10 * np.abs(X))

    def plot_template_matrix(self):
        fig = plt.figure(figsize=(4, 8))
        ax = fig.add_subplot(111)
        ax.imshow(self.W_init, cmap='gray', vmin=0, origin='lower', interpolation='nearest')
        plt.title('W_init')
        plt.show()

    def plot_activation_matrix(self):
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111)
        ax.imshow(self.H, cmap='gray', vmin=0, origin='lower', interpolation='nearest')
        plt.title('H')
        plt.show()

    def plot_instrument_spectra(self):
        for midi_note, instrument in self.instrument_codes.items():
            instrument.plot_recording()

    def plot_recording_spectrum(self):
        T_coef = np.arange(self.V.shape[1]) * self.hop / self.Fs
        F_coef = np.arange(self.V.shape[0]) * self.Fs / self.window
        left = min(T_coef)
        right = max(T_coef) + self.window / self.Fs
        lower = min(F_coef)
        upper = max(F_coef)
        plt.imshow(self.V, vmin=0, origin='lower', aspect='auto', cmap='gray_r', extent=[left, right, lower, upper])
        plt.xlabel('Time (seconds)')
        plt.ylabel('Frequency (Hz)')
        plt.title('Spectrogram')
        plt.show()