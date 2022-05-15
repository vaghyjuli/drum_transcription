import matplotlib.pyplot as plt
import librosa
import numpy as np

from Instrument import Instrument
from nmfd import *

class NMFLabels:
    def __init__(self, _wav_file, _instrument_codes, _nmf_type='NMFD'):
        self.wav_file = _wav_file
        self.instrument_codes = _instrument_codes
        self.window = 512
        self.hop = 256
        self.calculate_STFT()
        self.nmf_type = _nmf_type
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
            self.nmf(init=True)
        elif self.nmf_type == 'NMFD':
            self.nmfd()

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

    def nmf(self, init=False, norm=False, report=False):
        thresh = 0.001
        L = 1000
        R = len(self.instrument_codes)
        K = self.V.shape[0]
        N = self.V.shape[1]
        W = self.W_init if init else np.random.rand(K, R)
        H = np.random.rand(R, N)
        H_W_error = np.zeros((2, L))
        ell = 1
        below_thresh = False
        eps_machine = np.finfo(np.float32).eps
        while not below_thresh and ell <= L:
            H_ell = H
            W_ell = W
            H = H * (W.transpose().dot(self.V) / (W.transpose().dot(W).dot(H) + eps_machine))
            #W = W * (self.V.dot(H.transpose()) / (W.dot(H).dot(H.transpose()) + eps_machine))
            H_error = np.linalg.norm(H - H_ell, ord=2)
            W_error = np.linalg.norm(W - W_ell, ord=2)
            H_W_error[:, ell-1] = [H_error, W_error]
            if report:
                print('Iteration: ', ell, ', H_error: ', H_error, ', W_error: ', W_error)
            if H_error < thresh and W_error < thresh:
                below_thresh = True
                H_W_error = H_W_error[:, 0:ell]
            ell += 1
        if norm:
            for r in range(R):
                v_max = np.max(W[:, r])
                if v_max > 0:
                    W[:, r] = W[:, r] / v_max
                    H[r, :] = H[r, :] * v_max
        self.V_approx = W.dot(H)
        V_approx_err = np.linalg.norm(self.V - self.V_approx, ord=2)
        self.W = W
        self.H = H
        #print(f"NMF finished with V_approx_err={V_approx_err}\n")
        i = 0
        for midi_note, instrument in self.instrument_codes.items():
            instrument.set_activation(H[i])
            instrument.find_onsets()
            i+=1

    def nmfd(self):
        R = self.W_init.shape[1]
        K = self.V.shape[0]
        N = self.V.shape[1]
        T = self.W_init.shape[2]
        W, H, nmfdV, costFunc, tensorW = NMFD(V=self.V, T=T, R=R, W_init=self.W_init, fixW=True)
        i = 0
        for midi_note, instrument in self.instrument_codes.items():
            instrument.set_activation(H[i])
            instrument.find_onsets()
            i+=1