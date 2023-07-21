import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from common_functions import NamingClass
from common_settings import path_models

import glob
import os

from matplotlib.style import use


def plot_chart(file_path):
    # Read the csv file
    df = pd.read_csv(file_path)

    # Filter the DataFrame where "sess_eps" column is 0
    df = df[df['sess_eps'] == 0.0]

    # Create a new column "max_q" with the index of the max value across q1, q2, q3
    df['max_q'] = df[['q1', 'q2', 'q3']].idxmax(axis=1)
    df['max_q'] = df['max_q'].map({'q1': 1, 'q2': 2, 'q3': 3})

    # Get the unique agent_i values
    agents = df['agent_i'].unique()

    # Directory to save the plots
    # plot_dir = 'plots'
    # if not os.path.exists(plot_dir):
    #     os.makedirs(plot_dir)

    # Loop through each unique agent
    N = len(agents)
    plt.subplots(N, 1, figsize=(15, 20), dpi=200)
    for i, agent in enumerate(agents):
        # Filter the DataFrame for the current agent
        plt.subplot(N, 1, i + 1)
        df_agent = df[df['agent_i'] == agent]

        # Plot the "max_q" column
        df_agent['max_q'].plot(kind='line', alpha=0.5)
        plt.title(f'Agent {agent}')
        plt.xlabel('Index')
        plt.ylabel('Q arg')

        # Save the figure
        # plt.savefig(f'{plot_dir}/agent_{agent}.png')
        # plt.close()
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(file_path), f"gpt-plot-{os.path.basename(file_path)}.png"))
    plt.close()
    # plt.show()


if __name__ == '__main__':
    use('ggplot')

    folders = glob.glob(f"{path_models}*")
    folders = [fold for fold in folders if os.path.isdir(fold)]
    for cur_model_path in folders:
        # name = os.path.basename(cur_model_path)
        # print(f"name: {name}")
        try:
            naming = NamingClass.from_path(cur_model_path)
        except IndexError as err:
            print(f"Skipping results of folder: {cur_model_path} ")
            continue
        # print(naming)

        res_files = glob.glob(os.path.join(cur_model_path, "data", "") + "*")
        file_names = [os.path.basename(fil) for fil in res_files]
        dates = set()
        for file in file_names:
            if not file.endswith('.csv'):
                continue
            dt, tm, nm = file.split('-')
            dt_str = f"{dt}-{tm}"
            dates.add(dt_str)

        # print(f"Plotting: {naming.path}")
        for dt_str in dates:
            file_path = os.path.join(cur_model_path, "data", f"{dt_str}-qvals.csv")
            print(f"Path ({dt_str}): {file_path}")
            plot_chart(file_path)
