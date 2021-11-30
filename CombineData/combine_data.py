# Combine Data
#
# This file combines WeatherEMF data with DPIV Bee Count Data.
#
# It does this by determining the earliest and the latest date/time
# between the two given files, creates an array of dates spanning that
# time range, pairs up the WeatherEMF data with the Time Array, and then
# pairs up the DPIF Bee Count data with the TimeArray/WeatherEMF array.
#
# The DPIV Bee Motion Count data is only present for the hours between
# 7am and midnight since the bees aren't active during the night. However,
# the WeatherEMF data covers all 24 hours of the day. Bee Count values
# during the night are marked as NaN, and missing WeatherEMF entries are
# also marked as NaN.

import argparse
import csv
from datetime import datetime, timedelta
import fileinput
from math import nan
import os
import pandas as pd
import shutil


def obtain_start_and_end_times(weather_emf_filename, motion_counts_filename):
    """
    Determines the latest start time and earliest end time between
    the two given .csv files.
    """
    weather_emf_earliest = None
    weather_emf_latest = None
    with open(weather_emf_filename, newline="") as weather_emf_file:
        reader = csv.DictReader(weather_emf_file, delimiter=",")
        # Obtain the first timestamp of the file
        for row in reader:
            weather_emf_earliest = datetime.strptime(row["Time"], "%Y-%m-%d %H:%M:%S")
            break
        # Obtain the last timestamp in the file
        for row in reader:
            weather_emf_latest = datetime.strptime(row["Time"], "%Y-%m-%d %H:%M:%S")

    motion_count_earliest = None
    motion_count_latest = None
    with open(motion_counts_filename, newline="") as motion_count_file:
        reader = csv.DictReader(motion_count_file, delimiter=",")
        # Obtain the first timestamp of the file
        for row in reader:
            motion_count_earliest = datetime.strptime(row["Time"], "%Y-%m-%d %H:%M:%S")
            break
        # Obtain the last timestamp in the file
        for row in reader:
            motion_count_latest = datetime.strptime(row["Time"], "%Y-%m-%d %H:%M:%S")

    # Select the latest start time and earliest end time
    earliest = motion_count_earliest
    latest = motion_count_latest
    if weather_emf_earliest > motion_count_earliest:
        earliest = weather_emf_earliest
    if weather_emf_latest < motion_count_latest:
        latest = weather_emf_latest

    # Using the absolute earliest and absolute latest times resulted in a lot
    # of empty data records from the weather since the DPIV data started earlier.
    # This will be left here for reference just in case it's ever needed again.
    #
    # earliest = motion_count_earliest
    # latest = motion_count_latest
    # if weather_emf_earliest < motion_count_earliest:
    #     earliest = weather_emf_earliest
    # if weather_emf_latest > motion_count_latest:
    #     latest = weather_emf_latest

    print(f"Time range between the two files: {earliest} to {latest}")
    return earliest, latest


def create_base_data_list(start_time, end_time):
    """
    Creates a list of lists where each list contains
    the record number and a timestamp incremented by
    15 minutes from the previous. There is a 15 minute
    incremented timestamp for the range specified by
    the start_time and the end_time. The first list
    consists of a header for the columns.
    """
    time_list = []
    # Add the column header list
    time_list.append(["Record Number", "Time"])
    # time_list.append(["Time"])
    time_entry = start_time
    record_number = 1
    while time_entry <= end_time:
        # Add a list consisting of the record number and the 15 minute
        # incremented timestamp
        time_list.append([record_number, time_entry.strftime("%Y-%m-%d %H:%M:%S")])
        # time_list.append([time_entry.strftime("%Y-%m-%d %H:%M:%S")])
        record_number = record_number + 1
        time_entry = time_entry + timedelta(minutes=15)
    return time_list


def string_to_date(date_str):
    """
    Converts a string date into a datetime object.
    """
    return datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")


def pair_file_data_with_data_list(data_list, filename):
    """
    Add each row in the file to the list in the
    data_list with a matching timestamp.
    """
    with open(filename, newline="") as data_file:
        reader = csv.reader(data_file, delimiter=",")
        file_row = next(reader)  # Get the header row in the file

        # Determine where the Time columns are in each dataset
        file_time_idx = file_row.index("Time")
        base_time_idx = data_list[0].index("Time")

        data_list[0] += file_row[file_time_idx + 1 :]  # Add the headers
        file_row = next(reader)  # Go to the first data row
        missing_data = [nan for i in range(len(file_row) - 2)]
        end_of_file = False

        # Loop through the timestamps (skipping the header row)
        for index, row in enumerate(data_list[1:]):
            # If the end of the file was reached on the previous iteration, fill the rest
            # of the entries with nan
            if end_of_file:
                data_list[index + 1] += missing_data
                # Don't do the checks below
                continue

            # If the base time is ever greater than the file time, it must have been a
            # duplicate entry or a backwards time jump
            if string_to_date(row[base_time_idx]) > string_to_date(
                file_row[file_time_idx]
            ):
                wrong_order = True
                while wrong_order:
                    # Skip the row
                    try:
                        file_row = next(reader)
                    except StopIteration:
                        file_row = missing_data
                        end_of_file = True
                    wrong_order = string_to_date(row[base_time_idx]) > string_to_date(
                        file_row[file_time_idx]
                    )

            # Check if the data_list timestamp matches with a timestamp in the file
            if string_to_date(row[base_time_idx]) == string_to_date(
                file_row[file_time_idx]
            ):
                # Add the file row to the data_list row with the matching timestamp
                data_list[index + 1] += file_row[file_time_idx + 1 :]
                try:
                    file_row = next(reader)
                except StopIteration:
                    file_row = missing_data
                    end_of_file = True
            else:
                # Insert nans for all the values except the record number and time
                data_list[index + 1] += missing_data
    return data_list


def write_combined_data(combined_data_list, filename):
    """
    Write the combined data to a .csv file.
    """
    with open(filename, "w", newline="") as csvfile:
        writer = csv.writer(csvfile, delimiter=",")
        for row in combined_data_list:
            writer.writerow(row)


def delete_incorrect_record_num_column(filename):
    """
    Now that the Weather and EMR data has been
    expanded, the record numbers are incorrect and
    should be removed.
    Correct ones will be added later.
    """
    df = pd.read_csv(
        filename,
        header=0,
        index_col="Time",
        parse_dates=True,
        infer_datetime_format=True,
    )
    
    # The original 'Record Number' column gets renamed to
    # Record Number.1 when Pandas automatically adds a
    # column of the same name. Delete the old one since
    # the record numbers are off now.
    del df["Record Number.1"]
    
    # Write the corrected data to the file.
    df.to_csv(filename)


def main():
    """
    Pair the WeatherEMF data with the BeeMotionCount data.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-we",
        "--weather-emf-file",
        action="store",
        required=True,
        dest="weather_emf_file",
        help="The .csv file containing the preprocessed weather and EMF data.",
    )
    parser.add_argument(
        "-c",
        "--motion-count-file",
        action="store",
        required=True,
        dest="motion_count_file",
        help="The .csv file containing the preprocessed bee motion count data to "
        "be combined with the weather and EMF data.",
    )
    parser.add_argument(
        "-o",
        "--output-file",
        action="store",
        required=True,
        dest="output_file",
        help="The output .csv file containing the combined weather, EMF, and bee motion count data.",
    )
    args = parser.parse_args()

    print(f"Weather and EMF file: {args.weather_emf_file}")
    print(f"Bee Motion Counts file: {args.motion_count_file}")
    print(f"Combined output file: {args.output_file}\n")

    print("Obtaining the time range")
    start_time, end_time = obtain_start_and_end_times(
        args.weather_emf_file, args.motion_count_file
    )
    base_data_list = create_base_data_list(start_time, end_time)

    print("Combining the Weather and EMF data")
    combined_data_list = pair_file_data_with_data_list(
        base_data_list, args.weather_emf_file
    )

    print("Combining the Motion Count data")
    combined_data_list = pair_file_data_with_data_list(
        combined_data_list, args.motion_count_file
    )

    print(f"Writing the Combined Data File to {args.output_file}")
    write_combined_data(combined_data_list, args.output_file)

    print(f"Removing the incorrect record number column")
    delete_incorrect_record_num_column(args.output_file)

    print("Done.")


if __name__ == "__main__":
    main()
