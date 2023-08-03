import pandas as pd
import numpy as  np
import time

import os

from common_settings import path_data_folder


if __name__ == "__main__":
    input_filepath = os.path.join(path_data_folder, "obv_600.txt")
    output_filepath = os.path.join(path_data_folder, "test_updating.txt")

    dataframe = pd.read_csv(input_filepath)

    with open(output_filepath, "wt")as fp:
        fp.write(",".join(map(str, dataframe.columns)))
        fp.write("\n")

    with open(output_filepath, "at", buffering=1)as fp:
        for i in range(104300, len(dataframe)):
            time.sleep(np.random.random() * 0.1 + 0.01)
            ser = dataframe.iloc[i, :]
            fp.write(','.join(map(str, ser)))
            fp.write("\n")
            print(f"Writing sample: {i}")
