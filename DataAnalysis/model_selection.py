# Model Selectrion from Grid Search Results
#
# Performs the following on the grid search results:
# - Sorts the rows in increasing order by AIC score
# - Calculates the delta between the lowest AIC score and all others
#   and creates a column to hold the deltas.
# - Calculates the weight, or likelihood of each model out of all models.
#   This represents the probability that of each model that it is the best.
#
# The first row will represent the best performing model.
#
# Written by Daniel Hornberger - 2021

import argparse
import math
from matplotlib import pyplot as plt
import numpy as np
import os
import pandas as pd


def get_args():
    """
    Parse the args passed to the script by the user
    and return them.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input-file",
        action="store",
        required=False,
        dest="input_file",
        help="The file containing the results of the grid search.",
    )
    parser.add_argument(
        "-d",
        "--directory",
        action="store",
        required=False,
        dest="directory",
        help="The directory containing a series of files with grid search results.",
    )
    return parser.parse_args()

def perform_selection(input_file):
    """
    Performs the following on the grid search results:
    - Sorts the rows in increasing order by AIC score
    - Calculates the delta between the lowest AIC score and all others
      and creates a column to hold the deltas.
    - Calculates the weight, or likelihood of each model out of all models.
      This represents the probability that of each model that it is the best.

    The first row will represent the best performing model. 
    """
    df = pd.read_csv(input_file, header=0)
    print("Unsorted:")
    print(df.head())
    print("")

    # Sort the rows in increasing order by AIC score
    sorted_df = df.sort_values(by="aicc", ascending=True)
    print("Sorted:")
    print(sorted_df.head())
    print("")

    # Create a Delta column of the lowest AIC compared to ever other AIC score
    deltas = []
    # print(sorted_df.iloc[0])
    # print(sorted_df.iloc[0]['aicc'])
    min_aicc = sorted_df.iloc[0]["aicc"]
    for i, row in sorted_df.iterrows():
        deltas.append(row["aicc"] - min_aicc)
    sorted_df["delta"] = deltas

    print("Deltas:")
    print(sorted_df.head())
    print("")

    # Create a Weight column representing the probability that the model is the
    # best out of all others
    # w_i = e^((-1/2)*delta_i) / np.sum(e^((-1/2)*delta))
    # Note that all the w_is should sum up to 1
    delta_sum = np.sum(np.exp((-1 / 2) * sorted_df["delta"]))
    print(f"Sum of the Deltas: {delta_sum}")
    weights = []
    for i, row in sorted_df.iterrows():
        weights.append(np.exp((-1 / 2) * row["delta"]) / delta_sum)
    sorted_df["weight"] = weights
    print(f"Sum of Weights: {np.sum(weights)}")

    print("Deltas:")
    print(sorted_df.head())
    print("")

    # Add _model_selection to the original filename and write out the results
    output_filename = os.path.splitext(input_file)[0] + "_model_selection.csv"
    print(f"Output filename: {output_filename}")
    sorted_df.to_csv(output_filename, index=True)


def main():
    """
    The main driver for the program.
    """
    args = get_args()
    
    if args.input_file:
        perform_selection(args.input_file)
    elif args.directory:
        # Operate on every file in the specified directory
        files = os.listdir(args.directory)
        for file in files:
            print(f"Operating on {args.directory + '/' + file}")
            perform_selection(args.directory + "/" + file)


if __name__ == "__main__":
    main()
