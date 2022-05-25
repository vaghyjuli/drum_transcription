import numpy as np
from reader import read_data

nmf_types = ["NMFD", "NMF"]
#nmf_types = ["NMF"]
activation_inits = ["random", "uniform"]
fixW_options = ["adaptive", "semi", "fixed"]
#fixW_options = ["semi", "fixed"]
window_sizes = [512]

params = {}
for nmf_type in nmf_types:
    for fixW_option in fixW_options:
        for window_size in window_sizes:
            for activation_init in activation_inits:
                print("-" * 30)
                params["nmf_type"] = nmf_type
                params["fixW"] = fixW_option
                params["window"] = window_size
                params["hop"] = int(window_size/2)
                params["initH"] = activation_init

                print(params)

                data_folder = r'/Users/juliavaghy/Desktop/0--synth'
                samples = read_data(data_folder, params)
                #print(f"\nIncluded samples: {samples}\n")

                precision = np.zeros(len(samples))
                recall = np.zeros(len(samples))
                f_measure = np.zeros(len(samples))
                for idx in range(len(samples)):
                    precision[idx], recall[idx], f_measure[idx] = samples[idx].evaluate()

                print(f"\nF={round(np.mean(f_measure), 2)}±{round(np.std(f_measure), 2)}")
                print(f"P={round(np.mean(precision), 2)}±{round(np.std(precision), 2)}")
                print(f"R={round(np.mean(recall), 2)}±{round(np.std(recall), 2)}\n")


                for idx in range(len(samples)):
                    #if abs(np.mean(f_measure) - f_measure[idx]) >= np.std(f_measure):
                    print(f"Sample {samples[idx].dir}", end='\t')
                    print(f"F={round(f_measure[idx], 2)}\tP={round(precision[idx], 2)}\tR={round(recall[idx], 2)}")