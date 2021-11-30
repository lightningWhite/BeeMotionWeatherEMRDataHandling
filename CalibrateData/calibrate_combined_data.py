# Calibrate Combined Data
#
# If data is combined from two Weather and EMR Sensing Stations,
# the readings may differ from one sensor to another. These differences
# can artificially skew the data, making it unreliable and misleading.
# Sensor readings between two EMF-390 sensors have been observed to be
# much higher on one sensor than the other. Thus, when combining the
# data from one sensor with another, these differences need to be
# corrected so they are consistent with one another.
# This assumes that the stations are deployed to the same location.
#
# Station 2 was chosen to be the reference station since its
# values were higher and calibrating Station 1 up to Station 2 wouldn't
# truncate any values to zero. This can be changed though.
#
# To calibrate Station 1's values to be more consistent with Station 2's
# values, the following is done:
# 1. Calculate the line of best fit on the Station 2 data alone, giving
#    the slope of the line
#    * y = mx + b
# 2. Using the slope from the line of best fit, apply the following
#    function to each data point of Station 1:
#    Calibrated Value = (m * Record Number) + Original Value + offset / ratio
#
# Using the slope was needed due to the seasonality of the EMR data.
# Over time, the readings grew higher for both stations, but more
# so for Station 2. Since Station 1 is calibrated to Station 2,
# Station 1 needed to follow the same seasonality pattern.
# Adding the offset to the original value brings the data points up to
# the same starting offset as Station 2, and dividing the offset
# by the ratio of Station 2's standard deviation to Station 1's
# standard deviation places the narrower Station 1 data in the middle
# of the Station 2 data spread, rather than at the top.
#
# This script currently only supports calibrating data using two
# different stations.

import argparse
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


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
    parser.add_argument(
        "-o",
        "--output-directory",
        action="store",
        required=True,
        dest="output_directory",
        help="The directory where the data calibrated data " "file will be written",
    )
    return parser.parse_args()


def calibration_function(x, m, c):
    """
    Function to be used by curve_fit.
    x: The input
    m: The slope
    c: The intercept
    """
    return (x * m) + c


def main():
    """
    The main driver for the program.
    """
    args = get_args()

    df = pd.read_csv(
        args.input_file,
        header=0,
        index_col=["Time"],
        parse_dates=True,
        infer_datetime_format=True,
    )

    # Separate the data by Station ID
    grouped = df.groupby("Station ID")

    column_names = df.columns

    calib_values = {
        "avg_rf_watts": [],
        "avg_rf_watts_frequency": [],
        "peak_rf_watts": [],
        "frequency_of_rf_watts_peak": [],
        "peak_rf_watts_frequency": [],
        "watts_of_rf_watts_frequency_peak": [],
        "avg_rf_density": [],
        "avg_rf_density_frequency": [],
        "peak_rf_density": [],
        "frequency_of_rf_density_peak": [],
        "peak_rf_density_frequency": [],
        "density_of_fr_density_frequency_peak": [],
        "avg_total_density": [],
        "max_total_density": [],
        "avg_ef": [],
        "max_ef": [],
        "avg_emf": [],
        "max_emf": [],
    }

    # Determine line of best fit, and use the slope * Record Number + original value to get the new value
    popt, pcov = curve_fit(
        calibration_function,
        grouped.get_group(2.0)["Record Number"],
        grouped.get_group(2.0)["Avg. Total Density (W m^(-2))"].interpolate(),
    )
    print(f"Line of best fit: y = {popt[0]} * x + {popt[1]}")

    # Grab the slope from the returned coefficients
    slope = popt[0]
    intercept = popt[1]

    # Calculate the standard deviation for each station since the monitored range of each might be different
    station1_std = grouped.get_group(1.0).loc[:, "Avg. Total Density (W m^(-2))"].std()
    station2_std = grouped.get_group(2.0).loc[:, "Avg. Total Density (W m^(-2))"].std()

    print(f"Std. Deviation of Station 1 Data: {station1_std}")
    print(f"Std. Deviation of Station 2 Data: {station2_std}")

    # Spread ratio of Station 2 to Station 1
    ratio = station2_std / station1_std
    print(f"Station 2 to Station 1 Data Spread Ratio: {ratio}")

    # Calibrate each of the EMR data columns
    for date, row in df.iterrows():
        if row["Station ID"] == 1.0:
            # Calibrate the reading for Station 1 up to station 2
            # This applies the slope to the position of the record in the time series data
            # and then adds the original value to it and the Station 2 offset divided by the
            # ratio of Station 2 to Station 1's data spread. By dividing by the ratio,
            # it places the Station 1 data in the middle of Station 2, rather than at the top.
            calib_value = (
                row["Record Number"] * slope
                + row["Avg. Total Density (W m^(-2))"]
                + intercept / ratio
            )
            calib_values["avg_total_density"].append(calib_value)
        else:
            # Leave the reading unchanged for Station 2
            calib_values["avg_total_density"].append(
                row["Avg. Total Density (W m^(-2))"]
            )

    # Create a new DataFrame for the calibrated data columns
    df_calib = df.copy()
    df_calib["Avg. Total Density (W m^(-2))"] = calib_values["avg_total_density"]

    plt.figure()
    # Draw the original data
    plt.plot(df.index, df["Avg. Total Density (W m^(-2))"].interpolate())
    # Draw the line of best fit
    plt.plot(df.index, calibration_function(df["Record Number"], *popt))
    # Draw the corrected data
    plt.plot(df.index, df_calib["Avg. Total Density (W m^(-2))"].interpolate())
    plt.show()


if __name__ == "__main__":
    main()
