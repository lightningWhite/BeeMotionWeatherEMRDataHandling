# Data Utilities
#
# This file contains functions that perform various data preparations,
# manipulations, helper functions, etc. used by other Python modules.
#
# Written by Daniel Hornberger - 2021

import argparse
from datetime import datetime, timedelta
import math
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sn
from statistics import mean
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import sys


def smooth_column(df, column, window, center=False):
    """
    Performs a rolling average on the specified column.

    The default will be a tail-rolling average, however
    if center is set to True, it will set the labels
    at the center of the window. This function simply
    uses the pandas rolling method.

    Note: I don't think I'll use this. I don't like what it
          appears to do with the data.
    """
    new_col_name = column + "Smoothed"
    df[new_col_name] = df[column].rolling(window=window, center=center).mean()
    print(df[new_col_name].head(10))
    print("-----")
    print(df[column].head(10))

    df[column].plot()
    df[new_col_name].plot()
    plt.show()


def get_angle_average(angles):
    """
    Calculate the average angle from a list of angles.
    """
    sin_sum = 0.0
    cos_sum = 0.0

    for angle in angles:
        r = math.radians(angle)
        sin_sum += math.sin(r)
        cos_sum += math.cos(r)

    flen = float(len(angles))

    # Don't allow division by 0
    if flen == 0:
        return 0.0

    s = sin_sum / flen
    c = cos_sum / flen
    arc = math.degrees(math.atan(s / c))
    average = 0.0

    if s > 0 and c > 0:
        average = arc
    elif c < 0:
        average = arc + 180
    elif s < 0 and c > 0:
        average = arc + 360

    return 0.0 if average == 360 else average


def smartly_group_by_column(column, data, normalized=True):
    """
    Performs the appropriate method for grouping the
    data for the specified column. This includes returning
    the best precision for the column as well.

    Note: The Peak RF values and the Frequencies of those peaks are simply
    being averaged rather than having the max pulled out from them because
    the frequency would need to be associated with the correct RF value.
    In other words, the max frequency won't necessarily pertain to the
    max RF value. Thus, I've opted to do an average peak value over the
    time period, and an average frequency for the average peak value
    over the time period.
    """
    # Define the round amounts here so they can all be updated easily
    # if the data is normalized. We want to preserve decimal places
    # in this case.
    default_decimals = 1
    default_large_val_decimals = 0
    default_very_small_val_decimals = 16
    default_small_val_decimals = 3
   
    # Keep lots of decimal places since the values are all less than one
    # when normalized
    if normalized:
        default_decimals = default_very_small_val_decimals
        default_large_val_decimals = default_very_small_val_decimals
        default_very_small_val_decimals = default_very_small_val_decimals
        default_small_val_decimals = default_very_small_val_decimals

    if column == "Wind Direction (Degrees)":
        # Perform the correct average over 360 degrees
        return round(get_angle_average(data), default_decimals)
    if column == "Precipitation (Inches)":
        # Return the last value in the list since it's an accumulation metric
        return data[len(data) - 1]
    if column == "Wind Gust (MPH)" or "Max" in column:
        # Return the max value in the list
        return max(data)
    # Calculate the mean with a certain precision for the respective column
    if column == "Temperature (F)":
        return round(mean(data), default_decimals)
    if column == "Pressure (mbar)":
        return round(mean(data), default_decimals)
    if column == "Humidity (%)":
        return round(mean(data), default_decimals)
    if column == "Wind Speed (MPH)":
        return round(mean(data), default_decimals)
    if column == "Shortwave Radiation (W m^(-2))":
        return round(mean(data), default_large_val_decimals)
    if column == "Avg. RF Watts (W)":
        return round(mean(data), default_very_small_val_decimals)
    if column == "Avg. RF Watts Frequency (MHz)":
        return round(mean(data), default_large_val_decimals)
    if column == "Peak RF Watts (W)":
        return round(mean(data), default_very_small_val_decimals)
    if column == "Frequency of RF Watts Peak (MHz)":
        return round(mean(data), default_large_val_decimals)
    if column == "Peak RF Watts Frequency (MHz)":
        return round(mean(data), default_large_val_decimals)
    if column == "Watts of RF Watts Frequency Peak (W)":
        return round(mean(data), default_very_small_val_decimals)
    if column == "Avg. RF Density (W m^(-2))":
        return round(mean(data), default_very_small_val_decimals)
    if column == "Avg. RF Density Frequency (MHz)":
        return round(mean(data), default_large_val_decimals)
    if column == "Peak RF Density (W m^(-2))":
        return round(mean(data), default_very_small_val_decimals)
    if column == "Frequency of RF Density Peak (MHz)":
        return round(mean(data), default_large_val_decimals)
    if column == "Peak RF Density Frequency (MHz)":
        return round(mean(data), default_large_val_decimals)
    if column == "Density of RF Density Frequency Peak (W m^(-2))":
        return round(mean(data), default_very_small_val_decimals)
    if column == "Avg. Total Density (W m^(-2))":
        return round(mean(data), default_very_small_val_decimals)
    if column == "Max Total Density (W m^(-2))":
        return round(mean(data), default_very_small_val_decimals)
    if column == "Avg. EF (V/m)":
        return round(mean(data), default_decimals)
    if column == "Max EF (V/m)":
        return round(mean(data), default_decimals)
    if column == "Avg. EMF (mG)":
        return round(mean(data), default_decimals)
    if column == "Max EMF (mG)":
        return round(mean(data), default_decimals)
    if column == "UpwardMotion":
        return round(mean(data), default_large_val_decimals)
    if column == "DownwardMotion":
        return round(mean(data), default_large_val_decimals)
    if column == "LateralMotion":
        return round(mean(data), default_large_val_decimals)
    if column == "TotalMotion":
        return round(mean(data), default_large_val_decimals)
    if column == "Shortwave Radiation USU (MJ/m^2)":
        return round(mean(data), default_small_val_decimals)
    else:
        # print(f"An unaccounted for column was received that will be averaged: {column}")
        return round(mean(data), default_decimals)


def group_data(df, time_span, normalized=True):
    """
    Iterates through the dataframe and aggregates
    the rows into groups consisting of time ranges
    defined by time_span. Each column is aggregated
    appropriately for the type of data in that column.
    For instance, the TotalMotion column would average
    the values for the time_span, but the Precipitation
    column would be set to whatever value is highest.
    A new dataframe is returned.
    """
    print(f"Grouping the data into chunks spanning {time_span}")
    # Make a copy of the dataframe so as not to modify the original
    df_copy = df.copy()

    if time_span == timedelta(minutes=15):
        return df_copy
    elif time_span < timedelta(minutes=15):
        print("ERROR: The minimum grouping time_span is 15 minutes for this data.")
        sys.exit(0)

    # Remove the columns that won't aggregate well
    if "Record Number" in df_copy.columns:
        del df_copy["Record Number"]
    if "Wind Direction (String)" in df_copy.columns:
        del df_copy["Wind Direction (String)"]
    if "Station ID" in df_copy.columns:
        del df_copy["Station ID"]
    if "BeeMonitorID" in df_copy.columns:
        del df_copy["BeeMonitorID"]

    # Get the first row's time stamp
    start_time = df_copy.iloc[0].name

    # Get the time at the end of the time span
    end_time = start_time + time_span
    # If the time interval spans across the day, recenter it
    # so the time intervals will start from 00:00:00. This makes
    # it so a 24 hour period won't span across two different days
    # and only span across one whole day.
    if end_time.day != start_time.day:
        end_time = end_time.replace(hour=0, minute=0, second=0)

    # Get the final time in the dataset
    final_time = df_copy.iloc[len(df_copy) - 1].name

    # A list where each item is a list representing the newly averaged/grouped row
    new_rows = []
    index_times = []

    # For progress indication
    # counter = 0

    while end_time <= final_time:

        # Minus one minute from the end time so it's not included
        # Note that this may discard the very last data row
        segment = df_copy[start_time : end_time - timedelta(minutes=1)]
        
        # Check for data gaps resulting in an empty segment
        if segment.shape[0] < 1:
            start_time = end_time
            end_time = start_time + time_span
            # Skip this iteration
            continue

        # List to hold the newly grouped row values
        new_row = []

        # Add the last timestamp included in the grouping to be used as the
        # new index value. Note that 15 minutes are subtracted because the row
        # the end_time stamp isn't included in this iteration.
        index_times.append(end_time - timedelta(minutes=15))

        # Group each column appropriately according to the data it holds
        for column in segment.columns:
            new_value = smartly_group_by_column(column, list(segment[column]), normalized)
            new_row.append(new_value)

        new_rows.append(new_row)

        start_time = end_time
        end_time = start_time + time_span

        ## Progress indication
        # counter = counter + 1
        # if counter % 10 == 0:
        #    print(f"Processed {counter} groups of data")
        # if counter > 100:
        #    break

    # Create a new dataframe consisting of the newly grouped data
    new_df = pd.DataFrame(new_rows, columns=list(segment.columns), index=index_times)
    return new_df


def discretize_column(df, column_name, new_column_name, num_ranges):
    """
    Adds a new column to the dataframe where the column specified
    by column_name has its values categorized into the number of
    different ranges specified by num_ranges. These ranges would
    represent values such as low, med_low, med, med_high, high,
    for example. If num_ranges is -1, do a binary discretization.
    In other words, if it's 0, label it as zero; otherwise, label
    it as 1.
    TODO: This would be easier if I used the pandas 'cut' method.
    """
    if num_ranges != -1:
        print(f"Discretizing the {column_name} column into {num_ranges} ranges")
        minimum = df[column_name].min()
        # Get the range of the column so the label categories can be determined
        total_range = df[column_name].max() - df[column_name].min()
        print(f"Total range of the {column_name} column: {total_range}")
        label_range = total_range / num_ranges
        print(f"Label range of the {column_name} column: {label_range}")
    elif num_ranges == -1:
        print(f"Binarily discretizing the {column_name} column")
    else:
        print(f"Unsupported value specified for num_ranges: {num_ranges}")
        sys.exit(1)

    # A list to hold the values for the new range categories
    labels = []

    # Loop through each row in the dataframe
    for date, row in df.iterrows():
        value = row[column_name]

        # Binarily discretize the column
        if num_ranges == -1:
            if value == 0:
                labels.append(0)
            else:
                labels.append(1)
            # Skip the rest of the loop
            continue

        for label in range(num_ranges):
            lower = label * label_range
            upper = lower + label_range

            # Subtract minimum from it in case the minimum value isn't zero
            if (
                value - minimum >= lower
                and value - minimum < upper
                and label is not num_ranges - 1
            ):
                labels.append(label)
                # Don't check the rest now that it has been found
                break
            # Handle the last iteration so the values at the upper limit aren't missed
            elif (
                value - minimum >= lower
                and value - minimum <= upper + 1
                and label is num_ranges - 1
            ):
                labels.append(label)

    # Add the newly created labels column
    df[new_column_name] = labels


def remove_night_hours(series):
    """
    Removes the night hours from the Pandas
    dataframe series.
    """
    # The BeePi Monitor doesn't monitor the hives from 12am to 7am
    # However, bee activity frequently stops around 9pm, so I've set
    # to to that so the night hours don't affect the correlations as
    # much as with them included
    # start_night = "00:00"
    # start_night = "21:30"
    start_night = "21:00"
    end_night = "07:00"

    #### TEMPORARY
    # start_night = "21:00"
    # end_night = "08:00"
    ###

    print(f"Removing Night Hours: Excluding night timestamps between {start_night} and {end_night}")
    return series.between_time(end_night, start_night, include_end=False)


def add_col_month_and_hour(df):
    """
    Add a month column and an hour column to the dataframe.

    Since the time of year affects the bee behavior, adding
    a month column will help the decision trees fit a more
    accurate model. Adding the hour will also help fit a
    more accurate model because bees exhibit different levels
    of activity at different times of the day. They also
    aren't active during the night.
    """
    print("Adding a month column and an hour column")
    months = []
    hours = []
    # Loop through each row in the dataframe
    for date, row in df.iterrows():
        months.append(date.month)
        hours.append(date.hour)

    # Add the newly created labels column
    df["Month"] = months
    df["Hour"] = hours


def add_col_trend(df, column_name, interval):
    """
    Add a column that represents the trend in a specified
    column over a specified interval.

    The interval needs to be a timedelta object. The interval's
    minutes value MUST land on an even quarter hour (00, 15,
    30, 45).

    The column_name is the name of the column for which
    a trend column should be created.

    This function assumes that the dataframe has a timestamp
    as its index, and that every time stamp is on one of the
    quarters of the hour (00, 15, 30, 45). This allows the
    function to take advantage of index lookups for performance.
    """
    # Verify the interval is valid
    if (interval / timedelta(minutes=15)) % 1 != 0:
        print(
            "ERROR: add_col_trend requires that the interval be specified in 15 minute increments"
        )
        sys.exit(1)

    print(
        f"Adding a column to represent the trend of {column_name} over {interval} hour(s)"
    )
    trend_values = []

    # Loop through each row in the dataframe
    for date, row in df.iterrows():
        current_value = row[column_name]
        # Handle the first rows where the index won't exist as well
        # as any NaNs
        try:
            previous_value = df.loc[date - interval][column_name]
        except:
            # print("Invalid index encountered")
            # This will set the trend to 0, which should
            # cause the least amount of damage for missing values
            previous_value = current_value

        trend_values.append(round(current_value - previous_value, 2))

    # Add the newly created labels column
    new_col_name = column_name + f" {interval} Trend"
    df[new_col_name] = trend_values
    return new_col_name 
