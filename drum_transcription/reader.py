import glob
import os

from Sample import Sample
from Instrument import Instrument

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