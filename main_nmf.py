## Sources ##
# nmf : https://www.audiolabs-erlangen.de/resources/MIR/FMP/C8/C8S3_NMFbasic.html
# normalization : https://www.audiolabs-erlangen.de/resources/MIR/FMP/C3/C3S1_FeatureNormalization.html
# self-similarity matrix : https://www.audiolabs-erlangen.de/resources/MIR/FMP/C4/C4S2_SSM.html

import glob, os
import librosa
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

class Instrument:
    def __init__(self, file_name, path):
        self.name = file[:-4]
        self.N = 2048
        self.H = 512

        x, self.Fs = librosa.load(path)
        X = librosa.stft(x, n_fft=self.N, hop_length=self.H, win_length=self.N, window='hann', center=True, pad_mode='constant')
        self.Y = np.log(1 + 10 * np.abs(X))
        # take only the lower frequency part
        self.Y = self.Y[0:100,:]
        self.Y = normalize_feature_sequence(X=self.Y, norm="z")
        self.self_sim = np.dot(np.transpose(self.Y), self.Y)
        onset_lim = len(self.self_sim)
        for i in range(len(self.self_sim[0])):
            # ARBITRARY -> optimize
            if self.self_sim[0][i] < 60:
                onset_lim = i
                self.templates = [np.mean(self.Y[:, :onset_lim], axis=1), np.mean(self.Y[:, onset_lim:], axis=1)]
                return
        self.templates = [np.mean(self.Y, axis=1)]

    def plot_recording(self):
        T_coef = np.arange(self.Y.shape[1]) * self.H / self.Fs
        F_coef = np.arange(self.Y.shape[0]) * self.Fs / self.N
        plt.figure(figsize=(8, 2))
        plt.subplot(2, 2, 1)
        left = min(T_coef)
        right = max(T_coef) + self.N / self.Fs
        lower = min(F_coef)
        upper = max(F_coef)
        plt.imshow(self.Y, origin='lower', aspect='auto', cmap='gray_r', extent=[left, right, lower, upper])
        plt.xlabel('Time (seconds)')
        plt.ylabel('Frequency (Hz)')
        plt.title('Normalized spectrogram')
        ax = plt.subplot(2, 2, 2)
        plt.plot(F_coef, self.templates[0])
        plt.title('Template')
        ax = plt.subplot(2, 2, 3)
        plt.imshow(self.self_sim, origin='lower', aspect='auto', cmap='hot', vmin=0, interpolation='nearest', extent=[left, right, left, right])
        plt.xlabel('Time (seconds)')
        plt.ylabel('Time (seconds)')
        plt.title('Self-similarity matrix')
        plt.show()

    def plot_normalized(self):
        T_coef = np.arange(self.Y.shape[1]) * self.H / self.Fs
        F_coef = np.arange(self.Y.shape[0]) * self.Fs / self.N
        left = min(T_coef)
        right = max(T_coef) + self.N / self.Fs
        lower = min(F_coef)
        upper = max(F_coef)
        plt.figure(figsize=(8, 2))
        norm_options = ["1", "2", "max", "z"]
        for i in range(4):
            plt.subplot(3, 2, (i+1))
            plt.imshow(normalize_feature_sequence(X=self.Y, norm=norm_options[i]), origin='lower', aspect='auto', cmap='gray_r', extent=[left, right, lower, upper])
            plt.xlabel('Time (seconds)')
            plt.ylabel('Frequency (Hz)')
        ax = plt.subplot(3, 2, 5)
        plt.imshow(self.Y, origin='lower', aspect='auto', cmap='gray_r', extent=[left, right, lower, upper])
        plt.xlabel('Time (seconds)')
        plt.ylabel('Frequency (Hz)')
        plt.show()

    def get_templates(self):
        return self.templates
    
    def get_name(self):
        return self.name

def nmf(V, R, thresh=0.001, L=1000, W=None, H=None, norm=False, report=False):
    K = V.shape[0]
    N = V.shape[1]
    if W is None:
        W = np.random.rand(K, R)
    if H is None:
        H = np.random.rand(R, N)
    H_W_error = np.zeros((2, L))
    ell = 1
    below_thresh = False
    eps_machine = np.finfo(np.float32).eps
    while not below_thresh and ell <= L:
        H_ell = H
        W_ell = W
        H = H * (W.transpose().dot(V) / (W.transpose().dot(W).dot(H) + eps_machine))
        #W = W * (V.dot(H.transpose()) / (W.dot(H).dot(H.transpose()) + eps_machine))
        H_error = np.linalg.norm(H-H_ell, ord=2)
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
    V_approx = W.dot(H)
    V_approx_err = np.linalg.norm(V-V_approx, ord=2)
    return W, H, V_approx, V_approx_err, H_W_error

def normalize_feature_sequence(X, norm='2', threshold=0.0001, v=None):
    assert norm in ['1', '2', 'max', 'z']
    # Euclidian, Manhattan, Maximum, Feature normalization

    K, N = X.shape
    X_norm = np.zeros((K, N))

    if norm == '1':
        if v is None:
            v = np.ones(K, dtype=np.float64) / K
        for n in range(N):
            s = np.sum(np.abs(X[:, n]))
            if s > threshold:
                X_norm[:, n] = X[:, n] / s
            else:
                X_norm[:, n] = v

    if norm == '2':
        if v is None:
            v = np.ones(K, dtype=np.float64) / np.sqrt(K)
        for n in range(N):
            s = np.sqrt(np.sum(X[:, n] ** 2))
            if s > threshold:
                X_norm[:, n] = X[:, n] / s
            else:
                X_norm[:, n] = v

    if norm == 'max':
        if v is None:
            v = np.ones(K, dtype=np.float64)
        for n in range(N):
            s = np.max(np.abs(X[:, n]))
            if s > threshold:
                X_norm[:, n] = X[:, n] / s
            else:
                X_norm[:, n] = v

    if norm == 'z':
        if v is None:
            v = np.zeros(K, dtype=np.float64)
        for n in range(N):
            mu = np.sum(X[:, n]) / K
            sigma = np.sqrt(np.sum((X[:, n] - mu) ** 2) / (K - 1))
            if sigma > threshold:
                X_norm[:, n] = (X[:, n] - mu) / sigma
            else:
                X_norm[:, n] = v

    return X_norm

recording_file = "/Users/juliavaghy/Desktop/transcribe-this/recording.wav"

templates = []
instruments = []

folder_path = r'/Users/juliavaghy/Desktop/transcribe-this/instruments'
os.chdir(folder_path)
for file in glob.glob("*.wav"):
    file_path = os.path.join(folder_path, file)
    instruments.append(Instrument(file, file_path))

for instrument in instruments:
    #instrument.plot_recording()
    #instrument.plot_normalized()
    templates.append(instrument.get_templates())

templates_flat = [item for sublist in templates for item in sublist]

W_init = np.array(templates_flat, dtype=np.float64).transpose()
R = len(templates_flat)

N = 2048
hop = 512
x, Fs = librosa.load(recording_file)
X = librosa.stft(x, n_fft=N, hop_length=hop, win_length=N, window='hann', center=True, pad_mode='constant')
Y = np.log(1 + 10 * np.abs(X))
# take only the lower frequency part
V = Y[0:100,:]

W, H, V_approx, V_approx_err, H_W_error = nmf(V=V, R=R, W=W_init)

plt.imshow(H, cmap='hot', vmin=0, interpolation='nearest')
plt.title('H')
plt.show()

# plot H with different normalizations
"""
norm_options = ["1", "2", "max", "z"]
for i in range(4):
    plt.subplot(3, 2, (i+1))
    plt.imshow(normalize_feature_sequence(X=H, norm=norm_options[i]), cmap='hot', vmin=0, interpolation='nearest')
plt.subplot(3, 2, 5)
plt.imshow(H, cmap='hot', vmin=0, interpolation='nearest')
plt.show()
"""

onsets = []

# find onsets
# CHANGE RANGE IF MORE INSTRUMENTS ARE INCLUDED
#H = normalize_feature_sequence(X=H, norm="max")
Fs_feature = Fs / hop
colors = ["blue", "green", "cyan", "magenta", "yellow", "black", "white"]
for i in range(len(instruments)):
    # make first peaks visible
    activations = np.append([0, 0], H[i])
    T_coef = np.arange(len(H[i])) / Fs_feature
    time_unit = T_coef[1]
    T_coef = np.append([-time_unit*2, -time_unit], T_coef)

    for k in range(len(activations)):
        if(activations[k] < 0.1): activations[k] = 0
    plt.plot(T_coef, activations, color=colors[i])
    H_diff = np.diff(activations)
    #half-wave rectification
    for k in range(len(H_diff)):
        if H_diff[k] < 0.1:
            H_diff[k] = 0
    plt.plot(T_coef[:-1], H_diff, color=colors[i+1])
    peaks, properties = signal.find_peaks(H_diff, prominence=0.1)
    height = np.mean(H_diff[peaks])/2
    peaks, properties = signal.find_peaks(H_diff, height=height)
    peaks_sec = T_coef[peaks] + time_unit
    print(instruments[i].get_name(), peaks_sec)
    plt.title(instruments[i].get_name())
    plt.show()

    onsets.append(peaks_sec)

T_coef = np.arange(len(H[0])) / Fs_feature
for i in range(len(onsets)):
    plt.plot(onsets[i], [(i+1)*(100)]*len(onsets[i]), 'o', color=colors[i])
plt.ylim(top=(i+2)*(100))
plt.xlim(right=T_coef[-1])
plt.show()

# plot W and V
"""
fig = plt.figure()
plt.subplot(2, 2, 1)
plt.imshow(W_init, cmap='hot', interpolation='nearest')
plt.title('W_init')
plt.subplot(2, 2, 2)
plt.imshow(W, cmap='hot', interpolation='nearest')
plt.title('W')
plt.show()

fig = plt.figure()
plt.subplot(2, 2, 1)
plt.imshow(V, cmap='hot', interpolation='nearest')
plt.title('V')
plt.subplot(2, 2, 2)
plt.imshow(V_approx, cmap='hot', interpolation='nearest')
plt.title('V_approx')
plt.show()
"""