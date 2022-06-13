import numpy as np
from reader import read_data
import matplotlib.pyplot as plt
import csv
import os
from random import randint
from copy import deepcopy

DATA_FOLDER = r'/Users/juliavaghy/Desktop/0--data'
N_EXP = 1
data_file = 'data1/nonoise.csv'

# init params
params = {}
params["nmf_type"] = "NMF"
params["fixW"] = "fixed"
params["addedCompW"] = 0
params["beta"] = 4
params["window"] = 512
params["hop"] = int(params["window"]/2)
params["noise"] = "None"

nmf_types = ["NMF", "NMFD"]
#nmf_types = ["NMF"]
#fixW_options = ["adaptive", "semi", "fixed"]
fixW_options = ["semi"]
addedCompWs = [0]
betas = [0, float('inf'), 1, 2, 3, 4, 5, 6]
#noises = ["airplane", "chatter", "ambient", "mix"]
noises = ["None"]

while os.path.exists(data_file):
    data_file = 'data/' + str(randint(0, 100)) + '.csv'
print(f"Writing results in file {data_file}")

with open(data_file, 'w') as csv_file:
    header = [key for key, value in params.items()] + ["Sample", "F-mean", "P-mean", "R-mean", "F-sd", "P-sd", "R-sd"]
    writer = csv.writer(csv_file)
    writer.writerow(header)
    
    for fixW_option in fixW_options:
        params["fixW"] = fixW_option
        for beta in betas:
            params["beta"] = beta
            for addedCompW in addedCompWs:
                params["addedCompW"] = addedCompW
                for noise in noises:
                    params["noise"] = noise
                    for nmf_type in nmf_types:
                        params["nmf_type"] = nmf_type
                        print(params)

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

                        print(np.mean(f_measure).round(2))

                        for idx in range(len(samples)):
                            param_values = [value for key, value in params.items()] + [samples[idx].dir] + [meanax2(f_measure)[idx], meanax2(precision)[idx], meanax2(recall)[idx], stdax2(f_measure)[idx], stdax2(precision)[idx], stdax2(recall)[idx]]
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