from single_step_eval import start_backtest
from common_settings import path_data_folder

import tensorflow as tf
import os


if __name__ == "__main__":
    gpus = tf.config.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    # input_filepath = os.path.join(path_data_folder, "on_balance_volume.txt")  # local static
    input_filepath = os.path.join(path_data_folder, "obv_600.txt")  # local static
    # input_filepath = "/mnt/c/export/obv_600.txt"  # Machine file
    # input_filepath = os.path.join(path_data_folder, "test_updating.txt")  # Local file

    "MACHINE STARTS"
    start_backtest(
            input_filepath, time_window=50, time_feats=17, nodes=1000, arch_num=101, batch=300,
            loss='huber', lr='1e-05',  # output_filepath="/home/rafal/predict_101_huber_1e-05-it5.txt",
            iteration=5,
    )
