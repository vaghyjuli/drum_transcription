from reader import read_data

nmf_types = ["NMFD"]
fixW_options = [True, False]
window_sizes = [512]

params = {}
for nmf_type in nmf_types:
    for fixW_option in fixW_options:
        for window_size in window_sizes:
            print("-" * 30)
            params["nmf_type"] = nmf_type
            params["fixW"] = fixW_option
            params["window"] = window_size
            params["hop"] = int(window_size/2)

            print(params)

            data_folder = r'/Users/juliavaghy/Desktop/0-syth_data/data'
            samples = read_data(data_folder, params)
            print(f"\nIncluded samples: {samples}\n")
            tp_count = 0
            fp_count = 0
            fn_count = 0
            for sample in samples:
                tp_sample, fp_sample, fn_sample = sample.evaluate()
                tp_count += tp_sample
                fp_count += fp_sample
                fn_count += fn_sample

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