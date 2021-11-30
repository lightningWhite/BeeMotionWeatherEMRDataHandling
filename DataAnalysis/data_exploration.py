# Data Exploration
#
# This file contains functions that perform various data exploration techniques
# on a specified preprocessed and combined WeatherEMF and BeeMotionCount
# data file.
#
# Written by Daniel Hornberger - 2021

import argparse
from datetime import datetime
from matplotlib import pyplot
import math
import os
import pandas as pd
import seaborn as sn
from sklearn import preprocessing
import sys


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
        required=True,
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
        help="The directory where all of the data exploration "
        "results will be written",
    )
    parser.add_argument(
        "-ag",
        "--augment-data",
        action="store",
        required=False,
        dest="augment_data",
        help="Performs some data augmentation based off of the number "
        "specified. Valid numbers are defined in the 'data_augmentation'"
        "function. The first valid number is '1'.",
    )
    parser.add_argument(
        "-a",
        "--all",
        action="store_true",
        required=False,
        dest="all",
        help="Perform all of the data exploration techniques",
    )
    parser.add_argument(
        "-m",
        "--heat-maps",
        action="store_true",
        required=False,
        dest="heat_maps",
        help="Generate heat maps for all the data using Pearson's, "
        "Kendall's, and Tau's methods",
    )
    parser.add_argument(
        "-men",
        "--heat-maps-exclude-nights",
        action="store_true",
        required=False,
        dest="heat_maps_exclude_nights",
        help="Generate heat maps for all the data excluding night "
        "hours using Pearson's, Kendall's, and Tau's methods",
    )
    parser.add_argument(
        "-l",
        "--line-plots",
        action="store_true",
        required=False,
        dest="line_plots",
        help="Generate line plots for all the data for each column "
        "with the total bee motion count column vs time",
    )
    parser.add_argument(
        "-len",
        "--line-plots-exclude-nights",
        action="store_true",
        required=False,
        dest="line_plots_exclude_nights",
        help="Generate line plots for all the data excluding night "
        "hours for each column with the total bee motion count column vs time",
    )
    parser.add_argument(
        "-ltr",
        "--line-plots-time-range",
        action="store",
        required=False,
        dest="line_plots_time_range",
        help="Generate line plots for all the data for each column "
        "with the total bee motion count column vs time over a specified "
        "time range. The time range is specified as a string of two "
        "timestamps separated by a comma "
        "(e.g. '2020-05-16 16:45:00,2020-05-16 21:30:00')",
    )
    parser.add_argument(
        "-lentr",
        "--line-plots-exclude-nights-time-range",
        action="store",
        required=False,
        dest="line_plots_exclude_nights_time_range",
        help="Generate line plots for all the data excluding night "
        "hours for each column with the total bee motion count column vs time "
        "time range. The time range is specified as a string of two "
        "timestamps separated by a comma "
        "(e.g. '2020-05-16 16:45:00,2020-05-16 21:30:00')",
    )
    parser.add_argument(
        "-lc",
        "--list-columns",
        action="store_true",
        required=False,
        dest="list_columns",
        help="List all of the column headers of the data file",
    )
    parser.add_argument(
        "-c",
        "--custom-plot",
        action="store",
        required=False,
        dest="custom_plot",
        help="Generate a custom plot by specify a string of "
        "comma separated column headers to plot against time, and "
        "an optional time range separated by a comma can be specified "
        "following the column headers with '--'. "
        'For example: "Pressure (mbars),Precipitation (Inches),TotalMotion--2020-05-16 16:45:00,2020-05-16 21:30:00"',
    )
    parser.add_argument(
        "-cen",
        "--custom-plot-exclude-nights",
        action="store",
        required=False,
        dest="custom_plot_exclude_nights",
        help="Generate a custom plot by specify a string of "
        "comma separated column headers to plot against time, and "
        "an optional time range separated by a comma can be specified "
        "following the column headers with '--'. "
        'For example: "Pressure (mbars),Precipitation (Inches),TotalMotion--2020-05-16 16:45:00,2020-05-16 21:30:00"'
        "This option will exclude night hours.",
    )
    parser.add_argument(
        "-hccp",
        "--hard-coded-custom-plots",
        action="store",
        required=False,
        dest="hard_coded_custom_plots",
        help="Generates a set of custom plots with configurations "
        "hard-coded into the script. Different configuration sets can "
        "be specified by a number starting from '1'",
    )
    parser.add_argument(
        "-it",
        "--interactive",
        action="store_true",
        required=False,
        dest="interactive",
        help="Display the plots and keep them open for interaction. "
        "WARNING: Take care not to open too many plots at once since "
        "this can consume lots of memory. This is particularly pertinent "
        "when using the '--all' option.",
    )
    return parser.parse_args()


def sum_total_motion(series):
    """
    Augment the data by adding a column with the calculated
    accumulated sum of the bee TotalMotion for each day.
    This can be useful for viewing the overall total motion
    for a given day for comparisons, and it can also be useful
    to analyze the rate of bee motion throughout a day to see
    if certain events affect the rate of motion.
    """
    print(
        "Augmentation: Adding a column representing the accumulated "
        "total bee motion sum for each day."
    )
    sums = []
    previous_sum = 0
    for date, row in series.iterrows():
        # Reset the accumulated sum at the beginning of each day
        if date.hour == 0 and date.minute == 0:
            previous_sum = 0

        # Only update the sum if not dealing with missing data
        # Note that the BeePi monitor doesn't run during the night
        if not math.isnan(row["TotalMotion"]):
            new_sum = previous_sum + row["TotalMotion"]
            previous_sum = new_sum
            sums.append(new_sum)
        else:
            sums.append(previous_sum)

    # Add the newly created sum column
    series["TotalMotionSum"] = sums


def threshold_wind(series):
    """
    Set the wind values below a desired threshold to zero.
    This can be useful for determining at which point the
    wind speed begins to negatively affect bee motion
    (or determine if it actually does).
    """
    # Threshold in MPH since it's happening before it's normalized
    threshold = 3
    # series["TotalMotion"] = series["TotalMotion"].replace(["NaN"], 0)
    # series["Wind Speed (MPH)"] = series["Wind Speed (MPH)"].fillna(0, inplace=True)
    # print(series["Wind Gust (MPH)"].describe())
    # print(series.head())

    # series["Wind Speed (MPH)"] = series["Wind Speed (MPH)"].replace(math.nan, 0)
    series["Wind Speed (MPH)"].replace(math.nan, 0, inplace=True)
    # series["Wind Speed (MPH)"] = series.where(series["Wind Speed (MPH)"] > threshold, other=0)
    series["Wind Speed (MPH)"].mask(
        series["Wind Speed (MPH)"] <= threshold, 0, inplace=True
    )

    # print("After:")
    # print(series.tail())


def data_augmentation(series, selection):
    """
    Perform an augmentation operation on the data as specified
    by the selection value.
    """
    if selection == "1":
        sum_total_motion(series)
    elif selection == "2":
        threshold_wind(series)
    # elif selection == "3":
    #    remove_rf_density_outliers(series)
    else:
        print(
            "Invalid augmentation selection. No augmentation will be "
            "performed on the data."
        )


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
    start_night = "21:30"
    end_night = "07:00"

    #### TEMPORARY
    # start_night = "21:00"
    # end_night = "08:00"
    ###

    print(f"Excluding night timestamps between {start_night} and {end_night}")
    return series.between_time(end_night, start_night, include_end=False)


def generate_heat_maps(
    series, output_directory, exclude_nights=False, interactive=False
):
    """
    Generate heat maps representing the correlation
    between all of the data columns using
    Pearson's, Kendall's, and Tau's methods. All three
    methods are used to help prevent misinformation caused
    by outliers (to which Pearson's correlation coefficient
    is quite suceptible). It then saves the heat maps to
    the specified output directory.
    If exclude_nights is True, the night hours while the Bees
    are inactive are removed from the data so the correlations
    aren't skewed.
    """

    ## TEMPORARY: Remove this when done
    ## Possible TODO: Make it so I can generate correlation matrices for
    ## a time range.
    ## Create a correlation matrix for a single day
    # TODO: Comment this out!
    #print("IMPORTANT: Generating heat map for a single day.")
    #series = series.loc["2020-07-14 08:00:00" : "2020-07-14 20:00:00"]
    #print(series.describe())
    ######

    filename_suffix = ""
    base_save_path = output_directory + "/" + "correlation_heatmaps"
    if not os.path.exists(base_save_path):
        os.makedirs(base_save_path)

    if exclude_nights:
        filename_suffix = "_nights_excluded"
        series = remove_night_hours(series)

    # TODO: It would be nice to be able to specify a time range for the
    # correlation matrices
    # if time_range is not None:
    #    # Parse the start and end time
    #    start_end_times = time_range.split(",")

    #    # Grab only the data within the time range
    #    series = series.loc[start_end_times[0] : start_end_times[1]]

    #    # Add the time range to the filename
    #    filename_suffix = filename_suffix + "_" + time_range.replace(" ", "_").replace(",", "_to_")
    # series = series.loc["2020-05-19 09:00:00" : "2020-05-19 20:00:00"]

    # Remove the columns we don't care the find the correlation of
    del series["Record Number"]
    del series["BeeMonitorID"]

    # The different correlation calculation methods to apply to the data
    methods = ["pearson", "kendall", "spearman"]
    for method in methods:
        filename = method + "_heatmap" + filename_suffix + ".png"
        save_path = base_save_path + "/" + filename

        # Obtain a correlation matrix of the data using the specified method
        corr = series.corr(method=method)

        # Create a heatmap of the correlation matrix
        # Uncomment annot=True to get annotations
        pyplot.figure()
        sn.heatmap(
            corr,
            #annot=True,
            annot=False,
            xticklabels=1,
            yticklabels=1,
            linewidths=0.1,
            cmap="coolwarm",
        )

        # Save the generated heatmap of the correlation matrix
        pyplot.title(method + filename_suffix)
        pyplot.savefig(save_path, bbox_inches="tight", dpi=100)

        if not interactive:
            # Ensure that a new plot is created on the next iteration
            pyplot.close()
        else:
            # Show the Normalized line plots
            pyplot.show(block=False)

    # Keep the plots open until the user presses enter
    if interactive:
        blocker = input("Press the enter key to close all windows and end the script.")


def generate_line_plots(
    series,
    output_directory,
    exclude_nights=False,
    time_range=None,
    plot_columns=None,
    interactive=False,
):
    """
    Generate normalized line plots for all the data of each column
    individually plotted with the total bee motion count
    column vs time.
    :param exclude_nights If True, the night hours while the Bees
      are inactive are removed from the data so the correlations
      aren't skewed.
    :param time_range If specified, plots will be generated that
      only include rows from that time range. A time_range is
      specified a start and an end time separated by a column
      in a string format (e.g. '2020-05-16 16:45:00,2020-05-16 21:30:00')
    :param plot_columns If specified, a line plot will be generated
      plotting each specified column against time. plot_columns should
      be specified as a comma separated list of strings
      (e.g. "Pressure (mbars),Precipitation (Inches),TotalMotion")
    :param interactive Keep the graphs open for interaction until
      the user presses the enter key.
    """
    filename_suffix = ""
    base_save_path = output_directory + "/" + "line_plots"

    if exclude_nights:
        filename_suffix = "_nights_excluded"
        series = remove_night_hours(series)

    if plot_columns is not None:
        # Update the save directory indicating a custom plot
        base_save_path = base_save_path + "_custom"

    if time_range is not None:
        # Parse the start and end time
        start_end_times = time_range.split(",")

        # Grab only the data within the time range
        series = series.loc[start_end_times[0] : start_end_times[1]]

        # Add the time range to the filename
        filename_suffix = (
            filename_suffix + "_" + time_range.replace(" ", "_").replace(",", "_to_")
        )

    # Create the directory to hold the generated plots
    if not os.path.exists(base_save_path):
        os.makedirs(base_save_path)

    # Get the different columns to graph with the total bee motion counts
    if plot_columns is not None:
        columns = [plot_columns]
    else:
        columns = series.columns.tolist()
        # Remove the columns that shouldn't be graphed
        discard_columns = [
            "Record Number",
            "Wind Direction (String)",
            "Station ID",
            "BeeMonitorID",
            "UpwardMotion",
            "DownwardMotion",
            "LateralMotion",
            "TotalMotion",
        ]
        for column in discard_columns:
            columns.remove(column)

    # Generate the desired plots
    for column in columns:
        bee_motion_col = "TotalMotion"
        if plot_columns is None:
            filename = (
                column.replace(" ", "")
                .replace(".", "")
                .replace("/", "per")
                .replace(",", "And")
                + "And"
                + bee_motion_col
                + "_lineplot"
                + filename_suffix
                + ".png"
            )
        else:
            filename = (
                column.replace(" ", "")
                .replace(".", "")
                .replace("/", "per")
                .replace(",", "And")
                + "_lineplot"
                + filename_suffix
                + ".png"
            )
        save_path = base_save_path + "/" + filename

        if plot_columns is None:
            print(f"Plotting {column} and {bee_motion_col} vs Time")
        else:
            print(f"Plotting {column.replace(',',' & ')} vs Time")

        # Create a dataframe consisting of only the data we want to plot
        if plot_columns is None:
            line_plot_data = pd.DataFrame(series, columns=[bee_motion_col, column])
        else:
            line_plot_data = pd.DataFrame(series, columns=column.split(","))

        # Normalize the data so the columns can be usefully compared
        values = line_plot_data.values
        min_max_scaler = preprocessing.MinMaxScaler()
        values_scaled = min_max_scaler.fit_transform(values)

        # Create a new dataframe with the column names specified
        line_plot_data_normalized = pd.DataFrame(
            values_scaled, index=line_plot_data.index, columns=line_plot_data.columns
        )

        # Generate the line plot
        pyplot.figure()  # Use this if I want to open all of them at once and keep them open
        sn.lineplot(data=line_plot_data_normalized)
        # TODO: Remove this when done testing. I just want to see the actual values rather than the normalized ones for a test.
        # sn.lineplot(data=line_plot_data)

        # Save the generated line plot
        if plot_columns is None:
            pyplot.title(
                column
                + " & "
                + bee_motion_col
                + " vs Time"
                + filename_suffix.replace("_", " ")
            )
        else:
            pyplot.title(
                column.replace(",", " & ")
                + " vs Time"
                + filename_suffix.replace("_", " ")
            )
        pyplot.savefig(save_path, bbox_inches="tight", dpi=100)

        # Show the Normalized line plots
        # Use this if I want to open all of them at once and keep them open
        if interactive:
            pyplot.show(block=False)

        # Ensure that a new plot is created on the next iteration
        # Comment this out if its desired to keep the plots open
        if not interactive:
            pyplot.close()

    # This will keep the figures open until the user presses the enter key
    if interactive:
        blocker = input("Press the enter key to continue.")


def list_columns(series):
    """
    Lists the columns in the data to make it easier
    to know what column headers are valid when using
    the custom plots options.
    """
    columns = series.columns.tolist()
    for column in columns:
        print("\t" + column)


def get_hard_coded_custom_plot_set(selection):
    """
    Returns a set of hard coded custom plot configurations
    based on a selection number starting from '1'.
    """
    # Example: "Pressure (mbar),Precipitation (Inches),Wind Speed (MPH),TotalMotion--2020-07-10 07:00:00,2020-07-13 07:00:00",

    # Dictionary containing an entry for each configuration set
    custom_plots = {
        "1": [
            "Temperature (F),Pressure (mbar),TotalMotion--2020-07-10 07:00:00,2020-07-10 22:00:00",  # A day
            "Temperature (F),Pressure (mbar),TotalMotion--2020-07-10 07:00:00,2020-07-11 22:00:00",  # Two days
            "Temperature (F),Pressure (mbar),TotalMotion--2020-07-10 07:00:00,2020-07-12 22:00:00",  # Three days
            "Temperature (F),Pressure (mbar),TotalMotion--2020-07-10 07:00:00,2020-07-16 22:00:00",  # A week
            "Temperature (F),Pressure (mbar),TotalMotion--2020-07-10 07:00:00,2020-08-10 22:00:00",  # A month
        ],
        "2": [
            "Temperature (F),Avg. RF Watts (W),TotalMotion--2020-06-10 07:00:00,2020-06-10 22:00:00",  # A day
            "Temperature (F),Avg. RF Watts (W),TotalMotion--2020-06-10 07:00:00,2020-06-12 22:00:00",  # Three days
        ],
        "3": [
            "Temperature (F),Wind Speed (MPH),TotalMotion--2020-09-10 07:00:00,2020-09-10 22:00:00",  # A day
            "Temperature (F),Wind Speed (MPH),TotalMotion--2020-09-10 07:00:00,2020-09-12 22:00:00",  # Three days
        ],
        "4": [
            "Wind Speed (MPH),Pressure (mbar)--2020-09-10 07:00:00,2020-09-10 22:00:00",  # A day
            "Wind Speed (MPH),Pressure (mbar)--2020-09-10 07:00:00,2020-09-12 22:00:00",  # Three days
            "Wind Speed (MPH),Pressure (mbar)--2020-09-10 07:00:00,2020-09-16 22:00:00",  # A week
        ],
        "5": [
            "Precipitation (Inches),Pressure (mbar)--2020-09-10 07:00:00,2020-09-10 22:00:00",  # A day
            "Precipitation (Inches),Pressure (mbar)--2020-09-10 07:00:00,2020-09-16 22:00:00",  # A week
            "Precipitation (Inches),Pressure (mbar),TotalMotion--2020-08-10 07:00:00,2020-09-10 22:00:00",  # A month
        ],
    }
    return custom_plots[selection]


def generate_hard_coded_custom_plots(
    series, output_directory, selection, exclude_nights=False, interactive=False
):
    """
    A function containing predefined custom plot configurations.
    Each configuration can be selected by passing in an integer
    starting from '1'.
    """
    plot_set = get_hard_coded_custom_plot_set(selection)

    # Plot each custom plot in the selected plot set
    for plot in plot_set:
        # Split the comma separated columns from the time range
        columns_and_time_range = plot.split("--")
        columns = columns_and_time_range[0]
        if len(columns_and_time_range) > 1:
            time_range = columns_and_time_range[1]
        else:
            time_range = None

        generate_line_plots(
            series.copy(),
            output_directory,
            time_range=time_range,
            exclude_nights=exclude_nights,
            plot_columns=columns,
            interactive=interactive,
        )


def main():
    """
    The main driver of the program.
    """
    args = get_args()

    if (
        not args.all
        and not args.heat_maps
        and not args.heat_maps_exclude_nights
        and not args.line_plots
        and not args.line_plots_exclude_nights
        and not args.line_plots_time_range
        and not args.line_plots_exclude_nights_time_range
        and not args.custom_plot
        and not args.custom_plot_exclude_nights
        and not args.hard_coded_custom_plots
        and not args.list_columns
    ):
        print("WARNING: No arguments were specified. Nothing will be done.")
        sys.exit(0)

    # Load the specified data file
    # This will use the first row as the column headers and the
    # dates (second column) as the row labels
    series = pd.read_csv(
        args.input_file,
        header=0,
        index_col="Time",
        parse_dates=True,
        infer_datetime_format=True,
    )

    if args.augment_data:
        print("Augmenting the data...")
        # Don't pass a copy of series, we want it to modify the original
        data_augmentation(series, args.augment_data)

    if args.all or args.heat_maps:
        print("Generating heat maps...")
        generate_heat_maps(
            series.copy(), args.output_directory, interactive=args.interactive
        )

    if args.all or args.heat_maps_exclude_nights:
        print("Generating heat maps excluding night hours...")
        generate_heat_maps(
            series.copy(),
            args.output_directory,
            exclude_nights=True,
            interactive=args.interactive,
        )

    if args.all or args.line_plots:
        print("Generating line plots...")
        generate_line_plots(
            series.copy(), args.output_directory, interactive=args.interactive
        )

    if args.all or args.line_plots_exclude_nights:
        print("Generating line plots excluding night hours...")
        generate_line_plots(
            series.copy(),
            args.output_directory,
            exclude_nights=True,
            interactive=args.interactive,
        )

    if args.line_plots_time_range:
        print("Generating line plots for a specified time range...")
        generate_line_plots(
            series.copy(),
            args.output_directory,
            time_range=args.line_plots_time_range,
            interactive=args.interactive,
        )

    if args.line_plots_exclude_nights_time_range:
        print(
            "Generating line plots excluding night hours over a specified time range..."
        )
        generate_line_plots(
            series.copy(),
            args.output_directory,
            exclude_nights=True,
            time_range=args.line_plots_exclude_nights_time_range,
            interactive=args.interactive,
        )

    if args.custom_plot:
        print("Creating a custom plot...")
        # Split the comma separated columns from the time range
        columns_and_time_range = args.custom_plot.split("--")
        columns = columns_and_time_range[0]
        if len(columns_and_time_range) > 1:
            time_range = columns_and_time_range[1]
        else:
            time_range = None

        generate_line_plots(
            series.copy(),
            args.output_directory,
            time_range=time_range,
            plot_columns=columns,
            interactive=args.interactive,
        )

    if args.custom_plot_exclude_nights:
        print("Creating a custom plot with nights excluded...")
        # Split the comma separated columns from the time range
        columns_and_time_range = args.custom_plot_exclude_nights.split("--")
        columns = columns_and_time_range[0]
        if len(columns_and_time_range) > 1:
            time_range = columns_and_time_range[1]
        else:
            time_range = None

        generate_line_plots(
            series.copy(),
            args.output_directory,
            exclude_nights=True,
            time_range=time_range,
            plot_columns=columns,
            interactive=args.interactive,
        )

    if args.hard_coded_custom_plots:
        print(f"Generating hard-coded custom plots {args.hard_coded_custom_plots}...")
        generate_hard_coded_custom_plots(
            series,
            args.output_directory,
            args.hard_coded_custom_plots,
            interactive=args.interactive,
        )

    if args.list_columns:
        print("Listing the column headers of the data...")
        list_columns(series.copy())

    print("Done.")


if __name__ == "__main__":
    main()
