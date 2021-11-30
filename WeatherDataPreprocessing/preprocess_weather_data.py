# Preprocess Weather Data
#
# This file contains functions that can be used to clean up and preprocess
# the weather and EMF data. It removes unnecessary spaces after some of the
# commas in the CSV file, rounds the timestamps to the nearest quarter hour
# interval, and handles outliers.


import argparse
import csv
from datetime import datetime
from datetime import timedelta
import fileinput
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from scipy import stats
import shutil


def create_preprocessed_file(filename, preprocessed_filename):
    """
    Make a copy of the file to be preprocessed.
    """
    shutil.copyfile(filename, preprocessed_filename)
    print(f"A a new file has been created and named {preprocessed_filename}")


def remove_spaces_after_commas(filename):
    """
    Removes unnecessary spaces after commas in a specified csv file.
    """
    text_to_search = ", "
    replacement_text = ","
    print("Replacing ', ' with ','")
    with fileinput.FileInput(filename, inplace=True) as file:
        for line in file:
            print(line.replace(text_to_search, replacement_text), end="")


def round_timestamps(filename):
    """
    Round the timestamps in a specified csv file to the nearest quarter hour
    interval starting from the hour (00, 15, 30, 45 minute endings).
    """
    print("Rounding timestamps to the nearest quarter hour increment...")
    # Make a temporary copy of the data file to read
    temp_filename = ".temp_reader_file"
    shutil.copyfile(filename, temp_filename)
    with open(temp_filename, newline="") as temp_reader_file:
        reader = csv.DictReader(temp_reader_file, delimiter=",")

        with open(filename, "w", newline="") as csvfile_preprocessed:
            writer = csv.DictWriter(csvfile_preprocessed, fieldnames=reader.fieldnames)
            writer.writeheader()

            rounded_time = None
            previous = None
            zero_diff = 0
            thirty_diff = 0

            for row in reader:
                timestamp = datetime.strptime(row["Time"], "%Y-%m-%d %H:%M:%S.%f")

                # Round the time down
                floored_time = timestamp - timedelta(
                    minutes=timestamp.minute % 15,
                    seconds=timestamp.second,
                    microseconds=timestamp.microsecond,
                )

                # Get the difference between the actual time and the floored time
                time_diff = timestamp - floored_time

                # If the difference is over the half-way point, round upj
                if time_diff >= timedelta(minutes=7, seconds=30):
                    rounded_time = floored_time + timedelta(minutes=15)
                # Otherwise, round it down to the nearest 15 minute mark
                else:
                    rounded_time = floored_time

                # Write the row with the rounded time to the new csv file
                row["Time"] = rounded_time
                writer.writerow(row)

                # Count how many time gaps there are after rounding
                if previous:
                    if rounded_time - previous <= timedelta(minutes=0):
                        zero_diff += 1
                    elif rounded_time - previous > timedelta(minutes=15):
                        thirty_diff += 1

                # Update the previous time
                previous = rounded_time

            print(f"Duplicate Times: {zero_diff}")
            print(f"Time Gaps: {thirty_diff}")

    os.remove(temp_filename)


def handle_outliers(filename, show_plots):
    """
    Calculates the z-score (the number of standard deviations the point
    is from the mean) for each value in each EMR column. If the z-score is
    greater than 3, then the point is treated as an outlier.
    To handle the outlier, the points around it are used to interpolate a
    new value to replace it.
    Only the EMR values are searched for outliers, since some of the
    weather values which aren't outliers could unintentionally be removed
    by using the same process.

    If show_plots is true, the before and after plots will be shown for
    the columns when the outliers are removed.
    """
    df = pd.read_csv(
        filename,
        header=0,
        index_col="Time",
        parse_dates=True,
        infer_datetime_format=True,
    )

    for column in df.columns:
        # Correct the outliers in the EMR columns.
        # The weather columns are omitted because some non-outlier
        # values may be unintentionally removed.
        if column in [
            "Avg. RF Watts (W)",
            "Avg. RF Watts Frequency (MHz)",
            "Peak RF Watts (W)",
            "Frequency of RF Watts Peak (MHz)",
            "Peak RF Watts Frequency (MHz)",
            "Watts of RF Watts Frequency Peak (W)",
            "Avg. RF Density (W m^(-2))",
            "Avg. RF Density Frequency (MHz)",
            "Peak RF Density (W m^(-2))",
            "Frequency of RF Density Peak (MHz)",
            "Peak RF Density Frequency (MHz)",
            "Density of RF Density Frequency Peak (W m^(-2))",
            "Avg. Total Density (W m^(-2))",
            "Max Total Density (W m^(-2))",
            "Avg. EF (V/m)",
            "Max EF (V/m)",
            "Avg. EMF (mG)",
            "Max EMF (mG)",
        ]:
            print(f"Handling outliers: {column}")

            if show_plots:
                # Plot the data before outliers are removed
                plt.figure()
                plt.plot(
                    np.arange(len(df[column])), df[column].copy(), label="Original"
                )

            # Create a list of indexes where the value lies
            # outside +-3 standard deviations from the mean.
            # 99.7% of the data lies within 3 std deviations of the mean.
            outliers = (np.abs(stats.zscore(df[column]))) > 3

            # Set all outliers in the column equal to NaN
            df.loc[outliers, column] = np.nan

            # Interpolate the NaN values
            df[column].interpolate(inplace=True)

            if show_plots:
                # Show the before and after plots together
                plt.plot(np.arange(len(df[column])), df[column], label="Corrected")
                plt.title(column)
                plt.legend()
                plt.show()

                # Show only the corrected plot
                plt.figure()
                plt.plot(np.arange(len(df[column])), df[column], label="Corrected")
                plt.title(column + " Corrected Only")
                plt.legend()
                plt.show()

    # Write the corrected data to the file.
    df.to_csv(filename)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input-file",
        action="store",
        required=True,
        dest="input_file",
        help="The file to make a copy of and process",
    )
    parser.add_argument(
        "-o",
        "--output-file",
        action="store",
        required=True,
        dest="output_file",
        help="The name of the newly created processed file",
    )
    parser.add_argument(
        "-p",
        "--show-plots",
        action="store_true",
        required=False,
        dest="show_plots",
        help="Show the before and after plots when removing outliers",
    )

    args = parser.parse_args()

    create_preprocessed_file(args.input_file, args.output_file)
    remove_spaces_after_commas(args.output_file)
    round_timestamps(args.output_file)
    handle_outliers(args.output_file, args.show_plots)
    print("Done.")


if __name__ == "__main__":
    main()
