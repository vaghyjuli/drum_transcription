import glob
import os
import sys

data_folder = r'/Users/juliavaghy/Desktop/0-syth_data/data'
os.chdir(data_folder)

sample_directories = [item for item in os.listdir() if os.path.isdir(item)]

for sample_directory in sample_directories:

    os.chdir(os.path.join(data_folder, sample_directory))

    if len(glob.glob("*.mid")) != 1:
        print(f"There should be a single MIDI file in {sample_directory}")
        continue
    midi_file = os.path.join(data_folder, sample_directory, glob.glob("*.mid")[0])

    if len(glob.glob("*.wav")) != 1:
        print(f"There should be a single WAV file in {sample_directory}")
        continue
    wav_file = os.path.join(data_folder, sample_directory, glob.glob("*.wav")[0])

    with open ("info.txt", "r") as info_file:
        data = info_file.read().splitlines()
        n_instruments = len(data) - 4
        try:
            bpm = int(data[0].split()[0])
        except (ValueError, Exception) as error:
            print(f"Failed to identify BPM in {sample_directory}")
            continue

    os.chdir(os.path.join(data_folder, sample_directory, "instruments"))
    if os.listdir() != glob.glob("*.wav"):
        print(f"Strange files in {sample_directory}/instruments")
        continue
    if len(glob.glob("*.wav")) != n_instruments:
        print(f"According to info.txt, there should be {n_instruments} WAV files in {sample_directory}/instruments")
        continue
    instruments = [os.path.join(data_folder, sample_directory, recording) for recording in glob.glob("*.wav")]

    print(f"-> Folder {sample_directory} included")
    #print(bpm, midi_file, wav_file, instruments)