import glob
import os
import sys
import mido
from mido import MidiFile
from mido import Message
import matplotlib.pyplot as plt
import librosa
import numpy as np
from scipy import signal
from copy import deepcopy

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
        self.instrument_codes = _instrument_codes
        self.midi_labels = MIDILabels(_midi_file, _bpm, _instrument_codes)
        self.nmf_labels = NMFLabels(_wav_file, _instrument_codes)
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
        if comment:
            precision = tp_count / (tp_count + fp_count)
            recall = tp_count / (tp_count + fn_count)
            f_measure = (2*tp_count) / (2*tp_count + fp_count + fn_count)
            print(f"TP={tp_count}, FP={fp_count}, FN={fn_count}")
            print(f"precision = {precision}")
            print(f"recall = {recall}")
            print(f"F-measure = {f_measure}")
        return tp_count, fp_count, fn_count


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
        L = 1000
        R = self.W_init.shape[1]
        K = self.V.shape[0]
        N = self.V.shape[1]
        T = self.W_init.shape[2]
        W, H, nmfdV, costFunc, tensorW = NMFD(V=self.V, T=T, R=R, W_init=self.W_init, fixW=True)
        i = 0
        for midi_note, instrument in self.instrument_codes.items():
            instrument.set_activation(H[i])
            instrument.find_onsets(plot=True)
            i+=1

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
        self.window = 512
        self.hop = 256
        self.init_template()
        self.tp_count = 0    # true positives
        self.fp_count =0     # false positives
        self.fn_count = 0    # false negatives

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

    def init_template(self):
        x, self.Fs = librosa.load(self.wav_file)
        X = librosa.stft(x, n_fft=self.window, hop_length=self.hop, win_length=self.window, window='hann', center=True, pad_mode='constant')
        self.Y = np.log(1 + 10 * np.abs(X))
        self.template = np.mean(self.Y, axis=1)

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
            template2D = np.append(template2D, np.zeros((K, 1)), axis=1)

        if plot:
            T_coef = np.arange(template2D.shape[1]) * self.hop / self.Fs
            F_coef = np.arange(template2D.shape[0]) * self.Fs / self.window
            left = min(T_coef)
            right = max(T_coef) + self.window / self.Fs
            lower = min(F_coef)
            upper = max(F_coef)
            plt.imshow(template2D, vmin=0, origin='lower', aspect='auto', cmap='gray_r', extent=[left, right, lower, upper])
            plt.xlabel('Time (seconds)')
            plt.ylabel('Frequency (Hz)')
            plt.title('Spectrogram')
            plt.show()

        return template2D

    def plot_recording(self):
        T_coef = np.arange(self.Y.shape[1]) * self.hop / self.Fs
        F_coef = np.arange(self.Y.shape[0]) * self.Fs / self.window
        left = min(T_coef)
        right = max(T_coef) + self.window / self.Fs
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
        Fs_feature = self.Fs / self.hop
        T_coef = np.arange(len(self.activations)) / Fs_feature
        novelty = half_wave(np.append([0], np.diff(self.activations)))
        enhanced_novelty = half_wave(novelty - self.local_avg(novelty))
        peaks, properties = signal.find_peaks(enhanced_novelty, prominence=0.3)
        self.nmf_onsets = T_coef[peaks]

        if plot:
            plt.plot(T_coef, enhanced_novelty, color="blue")
            for onset in self.midi_onsets:
                plt.plot(onset*self.tick_duration, 0, 'o', color="red")
            plt.plot(self.nmf_onsets, [-0.1 for k in range(len(self.nmf_onsets))], 'o', color="black")
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
        print(f"{self.wav_file}\nTP={self.tp_count}\nFP={self.fp_count}\nFN={self.fn_count}\n")

    def local_avg(self, arr):
        avg_window = 3
        smoothed = np.zeros(arr.shape)
        padded = np.append(np.append([0]*avg_window, arr), [0]*(avg_window+1))
        for i in range(3, len(arr)+3):
            smoothed[i-3] = np.mean(padded[i-avg_window:i+avg_window+1])
        return smoothed


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

EPS = 2.0 ** -52

def NMFD(V, T, R, W_init, L=10, fixW=False):
    """
        Non-Negative Matrix Factor Deconvolution with Kullback-Leibler-Divergence
        and fixable components.

        Parameters
        ----------
        V: array-like
            Matrix that shall be decomposed (typically a magnitude spectrogram of dimension
            numBins x numFrames)

        L: int
            Number of NMFD iterations

        T: int
            Number of time frames for 2D-templates

        R: int
            Number of NMF components        

        Returns
        -------
        W: array-like
            List with the learned templates

        H: array-like
            Matrix with the learned activations

        nmfdV: array-like
            List with approximated component spectrograms

        costFunc: array-like
            The approximation quality per iteration

        tensorW: array-like
            If desired, we can also return the tensor
    """
    # use parameter nomenclature as in [2]
    K, N = V.shape
    initH = np.random.rand(R, N)
    tensorW = np.zeros((K, R, T))
    costFunc = np.zeros(L)

    # stack the templates into a tensor
    for r in range(R):
        tensorW[:, r, :] = W_init[:, r, :]

    # the activations are matrix shaped
    H = deepcopy(initH)

    # create helper matrix of all ones (denoted as J in eq (5,6) in [2])
    onesMatrix = np.ones((K, N))

    # this is important to prevent initial jumps in the divergence measure
    V_tmp = V / (EPS + V.sum())

    for iteration in range(L):
        # compute first approximation
        Lambda = convModel(tensorW, H)

        # store the divergence with respect to the target spectrogram
        costMat = V_tmp * np.log(1.0 + V_tmp/(Lambda+EPS)) - V_tmp + Lambda
        costFunc[iteration] = costMat.mean()

        # compute the ratio of the input to the model
        Q = V_tmp / (Lambda + EPS)

        # accumulate activation updates here
        multH = np.zeros((R, N))

        # go through all template frames
        for t in range(T):
            # use tau for shifting and t for indexing
            tau = deepcopy(t)

            # The update rule for W as given in eq. (5) in [2]
            # pre-compute intermediate, shifted and transposed activation matrix
            transpH = shiftOperator(H, tau).T

            # multiplicative update for W
            multW = Q @ transpH / (onesMatrix @ transpH + EPS)

            if not fixW:
                tensorW[:, :, t] *= multW

            # The update rule for W as given in eq. (6) in [2]
            # pre-compute intermediate matrix for basis functions W
            transpW = tensorW[:, :, t].T

            # compute update term for this tau
            addW = (transpW @ shiftOperator(Q, -tau)) / (transpW @ onesMatrix + EPS)

            # accumulate update term
            multH += addW

        # multiplicative update for H, with the average over all T template frames
        H *= multH / T

        # normalize templates to unit sum
        normVec = tensorW.sum(axis=2).sum(axis=0)

        tensorW *= 1.0 / (EPS+np.expand_dims(normVec, axis=1))

    W = list()
    nmfdV = list()

    # compute final output approximation
    for r in range(R):
        W.append(tensorW[:, r, :])
        nmfdV.append(convModel(np.expand_dims(tensorW[:, r, :], axis=1), np.expand_dims(H[r, :], axis=0)))

    return W, H, nmfdV, costFunc, tensorW


def convModel(W, H):
    """
        Convolutive NMF model implementing the eq. (4) from [2]. Note that it can
        also be used to compute the standard NMF model in case the number of time
        frames of the templates equals one.

        Parameters
        ----------
        W: array-like
            Tensor holding the spectral templates which can be interpreted as a set of
            spectrogram snippets with dimensions: numBins x numComp x numTemplateFrames

        H: array-like
            Corresponding activations with dimensions: numComponents x numTargetFrames

        Returns
        -------
        lamb: array-like
            Approximated spectrogram matrix
    """
    # the more explicit matrix multiplication will be used
    K, R, T = W.shape
    R, N = H.shape

    # initialize with zeros
    lamb = np.zeros((K, N))

    # this is doing the math as described in [2], eq (4)
    # the alternative conv2() method does not show speed advantages

    for k in range(T):
        multResult = W[:, :, k] @ shiftOperator(H, k)
        lamb += multResult

    lamb += EPS

    return lamb


def shiftOperator(A, shiftAmount):
    """
        Shift operator as described in eq. (5) from [2]. It shifts the columns
        of a matrix to the left or the right and fills undefined elements with
        zeros.

        Parameters
        ----------
        A: array-like
            Arbitrary matrix to undergo the shifting operation

        shiftAmount: int
            Positive numbers shift to the right, negative numbers
            shift to the left, zero leaves the matrix unchanged

        Returns
        -------
        shifted: array-like
            Result of this operation
    """
    # Get dimensions
    numRows, numCols = A.shape

    # Limit shift range
    shiftAmount = np.sign(shiftAmount) * min(abs(shiftAmount), numCols)

    # Apply circular shift along the column dimension
    shifted = np.roll(A, shiftAmount, axis=-1)

    if shiftAmount < 0:
        shifted[:, numCols + shiftAmount: numCols] = 0

    elif shiftAmount > 0:
        shifted[:, 0: shiftAmount] = 0

    else:
        pass

    return shifted

data_folder = r'/Users/juliavaghy/Desktop/0-syth_data/data'
samples = read_data(data_folder)
print(f"\nIncluded samples: {samples}\n")
tp_count = 0
fp_count = 0
fn_count = 0
for sample in samples:
    tp_sample, fp_sample, fn_sample = sample.evaluate()
    tp_count += tp_sample
    fp_count += fp_sample
    fn_count += fn_sample

if tp_count == 0:
    precision = 0
    recall = 0
    f_measure = 0
else:
    precision = tp_count / (tp_count + fp_count)
    recall = tp_count / (tp_count + fn_count)
    f_measure = (2*tp_count) / (2*tp_count + fp_count + fn_count)
print(f"TP={tp_count}, FP={fp_count}, FN={fn_count}")
print(f"precision = {precision}")
print(f"recall = {recall}")
print(f"F-measure = {f_measure}\n")
#print(Sample.__doc__)