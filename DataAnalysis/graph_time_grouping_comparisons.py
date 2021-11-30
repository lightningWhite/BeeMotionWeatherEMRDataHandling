# Graph Time Grouping Comparisons
#
# This file provides a simple function to graph the R^2 score
# and the confidence interval for each of the time grouping
# tests.
#
# Written by Daniel Hornberger
# 2021

import argparse
import matplotlib.pyplot as plt
import numpy as np
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
        help="The preprocessed and combined WeatherEMF and "
        "Bee Motion count data file",
    )
    return parser.parse_args()


def main():
    """
    The main driver for the program.
    """
    args = get_args()

    df = pd.read_csv(
        args.input_file,
        header=0,
        # This first version is for the unmodified file, the second is for pretty labels
        # index_col=["time_grouping_hrs"],
        index_col=["Time Grouping (Hours)"],
    )

    # Plot the R^2 score using the left y-axis
    fix, ax1 = plt.subplots()
    line1, = ax1.plot(df["R^2 Score"], color="tab:blue", label="R^2 Score")
    ax1.set_ylabel("R^2 Score")
    ax1.set_xlabel("Time Grouping (Hours)")
    #ax1.legend(loc="upper right")
    #ax1.legend()
    # ax1.grid(axis='x')

    # Create a twin object for two different y-axes on the same plot
    ax2 = ax1.twinx()
    # Plot the 95% confidence interval span on the right y-axis with a dashed line
    line2, = ax2.plot(df["95% Confidence Interval Span"], color="tab:orange", linestyle="--", label="95% Confidence Interval Span")
    ax2.set_ylabel("95% Confidence Interval Span")
    #ax2.legend(loc="lower right")
    ax2.legend([line1, line2], ["R^2 Score", "95% Confidence Interval Span"])
    plt.title("Time Grouping Comparison Results for the Random Forest Regressor")
    plt.grid(axis="x")
    plt.show()


if __name__ == "__main__":
    main()
