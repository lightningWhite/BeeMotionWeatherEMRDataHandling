# Add USU Solar Radiation
#
# This file takes the USU Solar Radiation data and adds it as a new column
# to the CombinedProcessedData file. The USU Solar Radiation file can be
# downloaded from https://climate.usu.edu/mchd/dashboard/dashboard.php?network=USUwx&station=1279257&units=E&showgraph=0&.
#
# In order to use this file with this script, it must be reformatted as
# follows:
#
#   1. Delete all columns except for "date_time" and "solarmj"
#   2. Rename "date_time" to "Time"
#   3. Rename "solarmj" to "Shortwave Radiation USU (MJ/m^2)"
#
# Since the data obtained from USU is in 1 hour increments, and the data
# collected by our weather stations is in 15 minute increments, the script
# will assign the 1 hour value to each 15 minute increment within the hour.
# It will also round up the hour value so the minutes aren't at 59.
#
# This script requires pandas to be installed. This can be done by running
# the following commands to create a Python virtual environment and then
# installing the package:
#
#   $ python3 -m venv env
#   $ source env/bin/activate
#   $ pip3 install --upgrade pip
#   $ pip3 install -r requirements.txt
#
# Written by Daniel Hornberger
# 2021

import argparse
from datetime import datetime, timedelta
import math
import pandas as pd
import sys


def get_args():
    """
    Parse the args passed to the script by the user
    and return them.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--data-file",
        action="store",
        required=True,
        dest="data_file",
        help="The preprocessed and combined WeatherEMF and "
        "Bee Motion count data file",
    )
    parser.add_argument(
        "-s",
        "--solar-radiation",
        action="store",
        required=True,
        dest="solar_radiation_file",
        help="The file containing the solar radiation data to be "
        "added to the preprocessed and combined WeatherEMF data file.",
    )
    parser.add_argument(
        "-o",
        "--output-file",
        action="store",
        required=True,
        dest="output_file",
        help="The name of the new .csv file containing the WeatherEMF "
        "data and the USU solar radiation data combined.",
    )

    return parser.parse_args()


def main():
    """
    The main driver for the program.
    """
    args = get_args()

    # Read in the Combined WeatherEMF data file
    df_data = pd.read_csv(
        args.data_file,
        header=0,
        index_col="Time",
        parse_dates=True,
        infer_datetime_format=True,
    )

    # Read in the USU Solar Radiation data file
    df_solar = pd.read_csv(
        args.solar_radiation_file,
        header=0,
        index_col="Time",
        parse_dates=True,
        infer_datetime_format=True,
    )

    solar_values = []
    progress = 0
    # Loop through each row of the data_file
    for index, row in df_data.iterrows():

        # Make sure the hour always has two digits
        hour = str(index.hour)
        if len(hour) < 2:
            hour = "0" + hour

        # Search the solar_radiation_file for the same date and hour
        # and grab the Solar Radiation value from the matching row
        query = df_solar[f"{index.date()} " + hour]
        if not query.empty:
            value = round(float(query["Shortwave Radiation USU (MJ/m^2)"]), 3)
        else:
            value = math.nan

        # Append the found value to the solar_values list
        solar_values.append(value)
        progress = progress + 1
        if progress % 100 == 0:
            print(f"{progress} rows processed")

    # Add the new column and write out the new csv file
    df_data["Shortwave Radiation USU (MJ/m^2)"] = solar_values
    df_data.to_csv(args.output_file)


if __name__ == "__main__":
    main()
