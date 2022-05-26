import numpy as np
from reader import read_data
import matplotlib.pyplot as plt

#nmf_types = ["NMFD", "NMF"]
nmf_types = ["NMF"]
#activation_inits = ["random", "uniform"]
activation_inits = ["uniform"]
#fixW_options = ["adaptive", "semi", "fixed"]
fixW_options = ["fixed"]
window_sizes = [512]
addedCompWs = range(10)

params = {}
F = []
for addedCompW in addedCompWs:
    params["addedCompW"] = addedCompW
    for nmf_type in nmf_types:
        params["nmf_type"] = nmf_type
        for fixW_option in fixW_options:
            params["fixW"] = fixW_option
            for window_size in window_sizes:
                params["window"] = window_size
                params["hop"] = int(window_size/2)
                for activation_init in activation_inits:
                    params["initH"] = activation_init
                    print("-" * 30)

                    print(params)

                    data_folder = r'/Users/juliavaghy/Desktop/0--synth'
                    samples = read_data(data_folder, params)
                    #print(f"\nIncluded samples: {samples}\n")

                    precision = np.zeros(len(samples))
                    recall = np.zeros(len(samples))
                    f_measure = np.zeros(len(samples))
                    for idx in range(len(samples)):
                        precision[idx], recall[idx], f_measure[idx] = samples[idx].evaluate()
                    
                    F.append(np.mean(f_measure))

                    print(f"\nF={round(np.mean(f_measure), 2)}±{round(np.std(f_measure), 2)}")
                    print(f"P={round(np.mean(precision), 2)}±{round(np.std(precision), 2)}")
                    print(f"R={round(np.mean(recall), 2)}±{round(np.std(recall), 2)}\n")


                    for idx in range(len(samples)):
                        #if abs(np.mean(f_measure) - f_measure[idx]) >= np.std(f_measure):
                        n_instruments = len(samples[idx].instrument_codes)
                        n_cropped = samples[idx].nmf_labels.n_cropped
                        #print(f"Sample {samples[idx].dir}", end='\t')
                        if f_measure[idx] > 0.9:
                            print(f"{n_cropped} / {n_instruments}", end='\t')
                            print(f"F={round(f_measure[idx], 2)}\tP={round(precision[idx], 2)}\tR={round(recall[idx], 2)}", end='\t')
                            print(f"Sam {samples[idx].dir}")

f, ax = plt.subplots(1)
ax.plot(addedCompWs, F)
ax.set_ylim(ymin=0, ymax=1)
plt.show()