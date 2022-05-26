import matplotlib.pyplot as plt
from scipy.io import wavfile
import os
import libfmp.c6
import librosa
from scipy import signal
import numpy as np

class Instrument:

    def __init__(self, _name, _recordings):
        self.name = _name
        self.recordings = _recordings
        self.N = 512
        self.H = 256
        self.Fs = []
        self.Y = []

        for fn_wav in self.recordings:
            x, Fs = librosa.load(fn_wav)
            self.Fs.append(Fs)
            X = librosa.stft(x, n_fft=self.N, hop_length=self.H, win_length=self.N, window='hanning')
            Y = np.log(1 + 10 * np.abs(X))
            self.Y.append(Y)

    def plot_recordings(self):
        for i in range(len(self.recordings)):
            T_coef = np.arange(self.Y[i].shape[1]) * self.H / self.Fs[i]
            F_coef = np.arange(self.Y[i].shape[0]) * self.Fs[i] / self.N
            print(F_coef)
            plt.figure(figsize=(8, 2))
            plt.subplot(1, 2, 1)
            left = min(T_coef)
            right = max(T_coef) + self.N / self.Fs[i]
            lower = min(F_coef)
            upper = max(F_coef)
            print(left, right, lower, upper)
            plt.imshow(self.Y[i], origin='lower', aspect='auto', cmap='gray_r', extent=[left, right, lower, upper])
            plt.xlabel('Time (seconds)')
            plt.ylabel('Frequency (Hz)')
            plt.ylim([0,5000])
            plt.tight_layout()
            ax = plt.subplot(1, 2, 2)
            summed_over_time = np.sum(self.Y[i], axis=1)
            plt.plot(F_coef, summed_over_time)
            plt.xlim([0,5000])
            plt.show()

## Metadata ##
colors = ["blue", "green", "cyan", "magenta", "yellow", "black", "white"]
directory = "/Users/juliavaghy/Desktop/DREANSS/dreanss_v1/bss_oracle/mix01/"
wav_f = directory + "drums.wav"
instruments = [Instrument(_name = "crash and bass", _recordings = [directory + "crash-bass-1.wav", directory + "crash-bass-2.wav"]),
                Instrument(_name = "hi hat", _recordings = [directory + "hi-1.wav", directory + "hi-2.wav"])]
all_files = os.listdir(directory)
txt_files = filter(lambda x: x[-4:] == '.txt', all_files)

for instrument in instruments:
    instrument.plot_recordings()

samplingFrequency, signalData = wavfile.read(wav_f)

x, Fs = librosa.load(wav_f)
x_duration = len(x)/Fs
nov, Fs_nov = libfmp.c6.compute_novelty_complex(x, Fs=Fs)
#nov, Fs_nov = libfmp.c6.compute_novelty_energy(x, Fs=Fs)
peaks, properties = signal.find_peaks(nov, prominence=0.02)
T_coef = np.arange(nov.shape[0]) / Fs_nov
peaks_sec = T_coef[peaks]

plt.figure(figsize=(8,2)) 
ax = plt.subplot(1,2,1)
plt.title('Spectrogram')    
Pxx, freqs, bins, im = plt.specgram(signalData, Fs=samplingFrequency, NFFT=512)
plt.xlabel('Time')
plt.ylabel('Frequency')
plt.ylim(0,10000)

onsets = set()
i=0
for txt_file in txt_files:
    file_name = directory + txt_file
    print(txt_file, colors[i])
    with open(file_name, "r") as f:
        for line in f:
            onset = float(line.split()[0])
            plt.plot(onset, (i+1)*(-200), 'o', color=colors[i])
            onsets.add(onset)
    i += 1
onsets = sorted(onsets)
#plt.plot(peaks_sec, [(i+1)*(-200) for k in range(len(peaks_sec))], 'ro', color="black")
plt.ylim(bottom=(i+2)*(-200))
#plt.xlim(0, 20)

ax = plt.subplot(1,2,2)
libfmp.b.plot_signal(nov, Fs_nov, ax=ax, color='k', title='Novelty function with detected peaks')
plt.plot(peaks_sec, nov[peaks], 'ro')

plt.show()

plt.plot(peaks_sec, [0.6 for i in range(len(peaks_sec))], 'ro', color="blue")
plt.plot(onsets,[0.5 for i in range(len(onsets))], 'ro', color="red")
"""
# for complex-based novelty
nov, Fs_nov = libfmp.c6.compute_novelty_complex(x, Fs=Fs)
peaks, properties = signal.find_peaks(nov, prominence=0.02)
T_coef = np.arange(nov.shape[0]) / Fs_nov
peaks_sec = T_coef[peaks]
plt.plot(peaks_sec, [0.55 for i in range(len(peaks_sec))], 'ro', color="blue")
"""
plt.ylim(0, 1)
plt.show()


print(f"{len(peaks)} peaks detected for {len(onsets)} onsets")
peak_idx = 0
tp_errors = []
for onset in onsets:
    # error for true positives, ignoring false positives as (ideally) they can later be filtered out
    tp_errors.append(min(abs(peaks_sec - onset)))
total_tp_error = sum(tp_errors)
avg_tp_error = total_tp_error / len(onsets)
max_tp_error = max(tp_errors)
print(f"total_tp_error = {total_tp_error} s = {total_tp_error * 1000} ms")
print(f"avg_tp_error = {avg_tp_error} s = {avg_tp_error * 1000} ms")
print(f"max_tp_error = {max_tp_error} s = {max_tp_error * 1000} ms")
