import numpy as np
from reader import read_data

nmf_types = ["NMFD", "NMF"]
activation_inits = ["random", "uniform"]
fixW_options = ["adaptive", "semi", "fixed"]
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

                data_folder = r'/Users/juliavaghy/Desktop/0-syth_data'
                samples = read_data(data_folder, params)
                print(f"\nIncluded samples: {samples}\n")
                tp_count = 0
                fp_count = 0
                fn_count = 0
                f_measures = []
                for sample in samples:
                    tp_sample, fp_sample, fn_sample, f_measure = sample.evaluate()
                    tp_count += tp_sample
                    fp_count += fp_sample
                    fn_count += fn_sample
                    f_measures.append(f_measure)

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

                mean_f_measure = sum(f_measures) / len(f_measures)
                sd_f_measure = np.std(f_measures)
                for idx in range(len(f_measures)):
                    if abs(mean_f_measure - f_measures[idx]) >= sd_f_measure:
                        print(f"F={f_measures[idx]} in {samples[idx].dir}")