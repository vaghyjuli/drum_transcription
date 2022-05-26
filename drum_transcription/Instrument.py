import matplotlib.pyplot as plt
import librosa
import numpy as np
from scipy import signal

EPS = 2.0 ** -52

class Instrument:
    """
        A class storing the onsets and MIDI note of a given instrument in a sample.
        Args:
            _midi_note (int): MIDI note of instrument.
            _color (str): Allocated color, to be used for plotting.
            _idx (int) : Allocated instrument index in the sample, to be used for plotting.
            _wav_file (str) : Path to the instrument's template recording.
            _window (int) : STFT window size
            _hop (int) : STFT hop size
        Attributes:
            midi_note (int): MIDI note of instrument.
            color (str): Allocated color, to be used for plotting.
            idx (int) : Allocated instrument index in the sample, to be used for plotting.
            onsets (ndarray int): Array containig the onset ticks of the instrument.
            wav_file (str) : Path to the instrument's template recording.
            window (int) : STFT window size
            hop (int) : STFT hop size
        Methods:
            print_onsets()s
            add_midi_onset()
            plot()
            compare()
    """
    def __init__(self, _midi_note, _color, _idx, _wav_file, _params):
        self.midi_note = _midi_note
        self.color = _color
        self.idx = _idx
        self.midi_onsets = [] # tick
        self.wav_file = _wav_file
        self.params = _params
        self.init_template()
        self.tp_count = 0    # true positives
        self.fp_count =0     # false positives
        self.fn_count = 0    # false negatives

        theta = {
            "NMF" : 3,
            "NMFD" : 4
        }
        self.THETA = theta[self.params["nmf_type"]]

    def __str__(self):
        return self.midi_note

    def print_onsets(self):
        """
            Print the instrument's MIDI onset ticks.
                Instrument MIDI note: [onset ticks]
        """
        print(f"{self.midi_note}: {self.midi_onsets}")

    def set_tick_duration(self, tick_duration):
        self.tick_duration = tick_duration

    def add_midi_onset(self, onset):
        self.midi_onsets.append(onset)

    def init_template(self, only_self_sim=True):
        x, self.Fs = librosa.load(self.wav_file)
        X = librosa.stft(x, n_fft=self.params["window"], hop_length=self.params["hop"], win_length=self.params["window"], window='hann', center=True, pad_mode='constant')
        self.Y = np.log(1 + 10 * np.abs(X))
        #self.Y = np.abs(X) + EPS
        if only_self_sim:
            normalized_Y = self.normalize(self.Y)
            self_sim = np.dot(np.transpose(normalized_Y), normalized_Y)
            onset_lim = len(self_sim[0])
            self.is_cropped = False
            for i in range(len(self_sim[0])):
                if self_sim[0][i] < 0.4:
                    onset_lim = i
                    self.is_cropped = True
                    break
            self.template = np.mean(self.Y[:, :onset_lim], axis=1)
        else:
            self.template = np.mean(self.Y, axis=1)

    def is_cropped(self):
        return self.is_cropped

    def normalize(self, X, threshold=0.0001):
        """
        Normalize feature sequence with respect to the Euclidian norm.
        """
        K, N = X.shape
        X_norm = np.zeros((K, N))
        v = np.ones(K, dtype=np.float64) / np.sqrt(K)
        for n in range(N):
            s = np.sqrt(np.sum(X[:, n] ** 2)) + EPS
            if s > threshold:
                X_norm[:, n] = X[:, n] / s
            else:
                X_norm[:, n] = v
        return X_norm + EPS

    def template_2D(self, T, plot=False):
        """
            For NMFD.
            Args
                T: int
                    num of timeframes in the 2D template
        """
        K = self.Y.shape[0]
        pad_len = T - self.Y.shape[1]
        template2D = self.Y
        for _ in range(pad_len):
            template2D = np.append(template2D, np.zeros((K, 1)) + EPS, axis=1)

        if plot:
            T_coef = np.arange(template2D.shape[1]) * self.params["hop"] / self.Fs
            F_coef = np.arange(template2D.shape[0]) * self.Fs / self.params["window"]
            left = min(T_coef)
            right = max(T_coef) + self.params["window"] / self.Fs
            lower = min(F_coef)
            upper = max(F_coef)
            plt.imshow(template2D, vmin=0, origin='lower', aspect='auto', cmap='gray_r', extent=[left, right, lower, upper])
            plt.xlabel('Time (seconds)')
            plt.ylabel('Frequency (Hz)')
            plt.title('Spectrogram')
            plt.show()

        return template2D

    def plot_recording(self):
        T_coef = np.arange(self.Y.shape[1]) * self.params["hop"] / self.Fs
        F_coef = np.arange(self.Y.shape[0]) * self.Fs / self.params["window"]
        left = min(T_coef)
        right = max(T_coef) + self.params["window"] / self.Fs
        lower = min(F_coef)
        upper = max(F_coef)
        plt.imshow(self.Y, vmin=0, origin='lower', aspect='auto', cmap='gray_r', extent=[left, right, lower, upper])
        plt.xlabel('Time (seconds)')
        plt.ylabel('Frequency (Hz)')
        plt.title('Spectrogram')
        plt.show()

    def plot_midi(self):
        for onset in self.midi_onsets:
            plt.plot(onset*self.tick_duration, (self.idx+1)*(200), 'o', color=self.color)
            # convert to seconds

    def set_activation(self, activations):
        self.activations = activations

    def find_onsets(self, plot=False):
        # half-wave rectification
        half_wave = lambda arr : (np.abs(arr) + arr) / 2
        Fs_feature = self.Fs / self.params["hop"]
        T_coef = np.arange(len(self.activations)) / Fs_feature
        #plt.plot(T_coef, self.activations, color="blue")
        novelty = half_wave(np.append([0], np.diff(self.activations)))
        #plt.plot(T_coef, novelty, color="yellow")
        enhanced_novelty = half_wave(novelty - self.local_avg(novelty))
        height = max(enhanced_novelty)/self.THETA
        #plt.plot(T_coef, [height] * len(T_coef), color="gray")
        peaks, properties = signal.find_peaks(enhanced_novelty, height=height)
        self.nmf_onsets = T_coef[peaks]

        if plot:
            plt.plot(T_coef, enhanced_novelty, color="orange")
            for onset in self.midi_onsets:
                plt.plot(onset*self.tick_duration, 0, 'o', color="red")
            plt.plot(self.nmf_onsets, [-height/10 for k in range(len(self.nmf_onsets))], 'o', color="black")
            plt.show()

    def evaluate(self):
        nmf_idx = 0
        for midi_idx in range(len(self.midi_onsets)):
            if nmf_idx >= len(self.nmf_onsets):
                self.fn_count += len(self.midi_onsets) - midi_idx
                break
            onset_sec = self.midi_onsets[midi_idx] * self.tick_duration
            distance = abs(onset_sec - self.nmf_onsets[nmf_idx])
            while (nmf_idx+1 < len(self.nmf_onsets)) and (abs(onset_sec - self.nmf_onsets[nmf_idx+1]) < distance):
                self.fp_count += 1
                nmf_idx += 1
                distance = abs(onset_sec - self.nmf_onsets[nmf_idx])
            if distance <= 0.05:
                self.tp_count += 1
                nmf_idx += 1
            else:
                self.fn_count += 1
        if nmf_idx < len(self.nmf_onsets)-1:
            self.fp_count += len(self.nmf_onsets) - nmf_idx - 1
        #print(f"{self.wav_file}\nTP={self.tp_count}\nFP={self.fp_count}\nFN={self.fn_count}\n")

    def local_avg(self, arr):
        avg_window = 3
        smoothed = np.zeros(arr.shape)
        padded = np.append(np.append([0]*avg_window, arr), [0]*(avg_window+1))
        for i in range(3, len(arr)+3):
            smoothed[i-3] = np.mean(padded[i-avg_window:i+avg_window+1])
        return smoothed