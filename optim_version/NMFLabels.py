import matplotlib.pyplot as plt
import librosa
import numpy as np
import random

from Instrument import Instrument
from nmfd import *
from nmf import *

EPS = 2.0 ** -52

class NMFLabels:
    def __init__(self, _params, _wav_file, _instrument_codes):
        self.wav_file = _wav_file
        self.instrument_codes = _instrument_codes
        self.params = _params
        self.calculate_STFT()
        self.initialize_template_matrix()
        self.factorize()
        self.Fs = 22050
        #self.plot_template_matrix()
        #self.plot_activation_matrix()

    def initialize_template_matrix(self):
        if self.params["nmf_type"] == 'NMF':
            templates = [instrument.template for midi_note, instrument in self.instrument_codes.items()]
            self.W_init = np.array(templates, dtype=np.float64).transpose()
        elif self.params["nmf_type"] == 'NMFD':
            T = max([instrument.Y.shape[1] for midi_note, instrument in self.instrument_codes.items()])
            templates = [instrument.template_2D(T) for midi_note, instrument in self.instrument_codes.items()]
            self.W_init = np.array(templates, dtype=np.float64).transpose((1, 0, 2))

    def factorize(self):
        if self.params["nmf_type"] == 'NMF':
            #cropped = [instrument.is_cropped for midi_note, instrument in self.instrument_codes.items()]
            #self.n_cropped = sum(cropped)
            V_approx, W, H = NMF(V=self.V, W_init=self.W_init, params=self.params)
        elif self.params["nmf_type"] == 'NMFD':
            V_approx, W, H = NMFD(V=self.V, W_init=self.W_init, params=self.params)
        i = 0
        for midi_note, instrument in self.instrument_codes.items():
            instrument.set_activation(H[i])
            instrument.find_onsets()
            i+=1

    def calculate_STFT(self):
        npy_file = self.wav_file[:-4] + f'-{self.params["window"]}.npy'
        self.V = np.load(npy_file, allow_pickle=True)
        if self.params["noise"] != "None":
            self.add_noise()
        self.V = np.log(1 + 10 * self.V)

    def add_noise(self):
        print("noisy")
        noise_file = f'/Users/juliavaghy/Desktop/0--data/background-loud/{self.params["noise"]}.wav'
        npy_file = noise_file[:-4] + f'-{self.params["window"]}.npy'
        noise = np.load(npy_file, allow_pickle=True)
        start = random.randint(0, noise.shape[1] - self.V.shape[1] - 1)
        end = start + self.V.shape[1]
        self.V = self.V + noise[:, start:end]
        
    def plot_recording_spectrum(self):
        T_coef = np.arange(self.V.shape[1]) * self.params["hop"] / self.Fs
        F_coef = np.arange(self.V.shape[0]) * self.Fs / self.params["window"]
        left = min(T_coef)
        right = max(T_coef) + self.params["window"] / self.Fs
        lower = min(F_coef)
        upper = max(F_coef)
        plt.imshow(self.V, vmin=0, origin='lower', aspect='auto', cmap='gray_r', extent=[left, right, lower, upper])
        plt.xlabel('Time (seconds)')
        plt.ylabel('Frequency (Hz)')
        plt.title('Spectrogram')
        plt.show()