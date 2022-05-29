import glob
import os
import librosa
import numpy as np

EPS = 2.0 ** -52

def stft_magnitude(wav_file):
    x, Fs = librosa.load(wav_file)
    X = librosa.stft(x, n_fft=1024, hop_length=512, win_length=1024, window='hann', center=True, pad_mode='constant')
    Y = np.abs(X) + EPS
    return Y

data_folder = "/Users/juliavaghy/Desktop/0--data"
os.chdir(os.path.join(data_folder, "data"))
sample_directories = [item for item in os.listdir() if os.path.isdir(item)]
for sample_directory in sample_directories:
    os.chdir(os.path.join(data_folder, "data", sample_directory))
    make_path = lambda f : os.path.join(data_folder, "data", sample_directory, f)
    if len(glob.glob("*.wav")) != 1:
        print(f"There should be a single WAV file in {sample_directory}")
        continue
    wav_file = os.path.join(data_folder, "data", sample_directory, (glob.glob("*.wav")[0]))
    stft = stft_magnitude(wav_file)
    np.save(f'{glob.glob("*.wav")[0][:-4]}.npy', stft)

os.chdir(os.path.join(data_folder, "kits"))
kits = [item for item in os.listdir() if os.path.isdir(item)]
for kit in kits:
    os.chdir(os.path.join(data_folder, "kits", kit, "instruments"))
    instruments = glob.glob("*.wav")
    for instrument in instruments:
        wav_file = os.path.join(data_folder, "kits", kit, "instruments", instrument)
        stft = stft_magnitude(wav_file)
        np.save(f'{instrument[:-4]}.npy', stft)

os.chdir(os.path.join(data_folder, "background"))
noises = glob.glob("*.wav")
for noise in noises:
    wav_file = os.path.join(data_folder, "background", noise)
    stft = stft_magnitude(wav_file)
    np.save(f'{noise[:-4]}.npy', stft)

#data = np.load(f'{noise[:-4]}.npy', allow_pickle=True)