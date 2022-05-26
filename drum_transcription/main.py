import numpy as np
from reader import read_data
import matplotlib.pyplot as plt
import csv
import os
from random import randint

DATA_FOLDER = r'/Users/juliavaghy/Desktop/0--synth'
N_EXP = 2
data_file = 'data.csv'

# init params
params = {}
params["addedCompW"] = 0
params["nmf_type"] = "uniform"
params["fixW"] = "fixed"
params["window"] = 512
params["hop"] = int(params["window"]/2)
params["initH"] = "uniform"

nmf_types = ["NMF", "NMFD"]
#nmf_types = ["NMF"]
activation_inits = ["random", "uniform"]
#activation_inits = ["uniform"]
fixW_options = ["adaptive", "semi", "fixed"]
#fixW_options = ["fixed"]
window_sizes = [256, 512, 1024, 2048]
#window_sizes = [512]
addedCompWs = range(5)

while os.path.exists(data_file):
    data_file = str(randint(0, 100)) + '.csv'
print(f"Writing results in file {data_file}")

with open(data_file, 'w') as csv_file:
    header = [key for key, value in params.items()] + ["F-mean", "P-mean", "R-mean", "F-sd", "P-sd", "R-sd"]
    writer = csv.writer(csv_file)
    writer.writerow(header)
    for nmf_type in nmf_types:
        params["nmf_type"] = nmf_type
        for fixW_option in fixW_options:
            params["fixW"] = fixW_option
            for window_size in window_sizes:
                params["window"] = window_size
                params["hop"] = int(window_size/2)
                for activation_init in activation_inits:
                    params["initH"] = activation_init

                    samples = read_data(DATA_FOLDER, params)
                    precision = np.zeros((N_EXP, len(samples)))
                    recall = np.zeros((N_EXP, len(samples)))
                    f_measure = np.zeros((N_EXP, len(samples)))

                    for exp_idx in range(N_EXP):

                        samples = read_data(DATA_FOLDER, params)
                        for sample_idx in range(len(samples)):
                            precision[exp_idx][sample_idx], recall[exp_idx][sample_idx], f_measure[exp_idx][sample_idx] = samples[sample_idx].evaluate()

                    mean2 = lambda a : np.mean(a).round(2)
                    meanax2 = lambda a: np.mean(a, axis=0).round(2)
                    std2 = lambda a : np.std(a).round(2)
                    stdax2 = lambda a: np.std(a, axis=0).round(2)

                    for idx in range(len(samples)):
                        param_values = [value for key, value in params.items()] + [meanax2(f_measure)[idx], meanax2(precision)[idx], meanax2(recall)[idx], stdax2(f_measure)[idx], stdax2(precision)[idx], stdax2(recall)[idx]]
                        writer.writerow(param_values)

                    #param_values = [value for key, value in params.items()] + [mean2(f_measure), mean2(precision), mean2(recall), std2(f_measure), std2(precision), std2(recall)]
                    #writer.writerow(param_values)

"""
precision_samples = np.mean(precision, axis=0)
#precision_total = np.mean(precision)
recall_samples = np.mean(recall, axis=0)
#recall_total = np.mean(recall)
f_measure_samples = np.mean(f_measure, axis=0)
#f_measure_total = np.mean(f_measure)
print(f_measure_samples)
"""
"""
for idx in range(len(samples)):
    #if abs(np.mean(f_measure) - f_measure[idx]) >= np.std(f_measure):
    #n_instruments = len(samples[idx].instrument_codes)
    #n_cropped = samples[idx].nmf_labels.n_cropped
    #print(f"Sample {samples[idx].dir}", end='\t')
    #if f_measure[idx] > 0.9:
        #print(f"{n_cropped} / {n_instruments}", end='\t')
    print(f"F={round(f_measure[idx], 2)}\tP={round(precision[idx], 2)}\tR={round(recall[idx], 2)}", end='\t')
    print(f"#{samples[idx].dir}")
"""

"""
f, ax = plt.subplots(1)
ax.plot(addedCompWs, F)
ax.set_ylim(ymin=0, ymax=1)
plt.show()
"""