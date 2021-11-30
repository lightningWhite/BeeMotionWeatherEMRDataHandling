# Random Forest Regressor
#
# An implementation of a Random Forest Regressor
# for predicting the Bee Motion from Weather and/or
# Electromagnetic Radiation. The regressor takes the
# mean of the decision made by each tree in the forest
# rather than the mode. This makes it suitable for regression.
#
# Written by Daniel Hornberger - 2021

# The process of running this script to determine the best combination
# of base columns to use, trend columns to use, and model
# hyperparameters to use is as follows:
#
# 1. Use the --orig-col-grid-search option to search the best base
#    columns to use. This will take around 7 hours to run.
#    1. Use the model_selection.py script to place the best performing
#       model at the top of the file. Use the columns of this result
#       as the columns_to_use in the trend_column_additive_analysis()
#       function. These values are hard-coded, so the user needs to
#       set them accordingly.
# 2. Use the --trend-column-additive-analysis option to determine the
#    trend intervals that produce the best model with the base columns.
#    1. Use the model_selection.py script on each file to place the best
#       performing model at the top of the file. Use the columns of these
#       results in the add_trend_columns() function in the else portion
#       to generate only these columns for the trend_column_grid_search()
#       function. These values are hard-coded.
# 3. Use the --trend-column-grid-search option to determine the best
#    combination of trend columns to use in conjunction with the base
#    columns.
#    1. Use the model_selection.py script to place the best performing
#       model at the top of the file.
#    2. Place top base model row from the selected model file from the
#       base column grid search at the top of the raw trend column grid
#       search results file, and run the model_selection.py script to
#       verify that the top model including trend columns is better
#       than the base columns model. Use the columns of this result
#       in the hyperparameter_grid_search() function in the
#       columns_to_use variable. These values are hard-coded.
# 4. Use the --hyperparameter-grid-search option to determine the best
#    Random Forest Regressor hyperparameters to use to obtain the best
#    performing model.
#    1. Use the model_selection.py script to place the best performing
#       model at the top of the file. Use the hyperparameters of the
#       top performing model as the default values of the
#       train_and_test_models() function.
# 5. Use the --time-grouping-tests option to produce the performance
#    results for each time grouping on the best obtained model.
#    1. Be sure to use the columns_to_use from the
#       grid_search_function in the time_grouping_tests() function.
#       Also, be sure to use the best Random Forest Hyperparameters.
#    2. Also, update the best_model() function accordingly.

import argparse
import copy
from datetime import datetime, timedelta
from itertools import combinations
import math
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import pickle
import os
import scipy.stats as stats
import seaborn as sn
from statistics import mean
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.feature_selection import RFECV
from sklearn.feature_selection import RFE
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
import sys

# Imports for Partial Least Squares Regression
from sklearn.cross_decomposition import PLSRegression

# Imports for K-Nearest-Neighbors
from sklearn.neighbors import KNeighborsRegressor

# Custom modules
import shared.util as util


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
    # Disabling this feature since I will likely run on a single hive.
    # If this is re-enabled, be sure to uncomment the lines that normalize
    # the TotalMotion column. They are currently commented out.
    # parser.add_argument(
    #    "-l",
    #    "--input-file-list",
    #    action="store",
    #    required=False,
    #    dest="input_file_list",
    #    help="The file path to a file containing a filename on each line "
    #    "that should be combined together. This is particularly useful when "
    #    "it's desirable to combine the data from multiple hives. Note that "
    #    "if --input-file is specified, it will override this argument.",
    # )
    parser.add_argument(
        "-tr",
        "--time-range",
        action="store",
        required=False,
        dest="time_range",
        help="The time range of the data to use for training and testing "
        "the model. The time range is specified as a string of two "
        "timestamps separated by a comma "
        "(e.g. '2020-05-17 00:00:00,2020-06-17 00:00:00')",
    )
    parser.add_argument(
        "-ocgs",
        "--orig-col-grid-search",
        action="store",
        required=False,
        dest="orig_col_grid_search",
        help="Specify the output directory where the file containing the "
        "results will be written for a grid search on selected original "
        "columns of the data. Note that this will likely take several hours "
        "to complete. If this is not specified, the grid search won't be performed. "
        "If no directory path is specified, the file will be written to the "
        "current directory.",
    )
    parser.add_argument(
        "-ocgsen",
        "--orig-col-grid-search-nights-excluded",
        action="store",
        required=False,
        dest="orig_col_grid_search_nights_excluded",
        help="Specify the output directory where the file containing the "
        "results will be written for a grid search on selected original "
        "columns of the data with the night hours removed. Note that this "
        "will likely take several hours to complete. If this is not specified, "
        "the grid search won't be performed. If no directory path is specified, "
        "the file will be written to the current directory.",
    )
    parser.add_argument(
        "-tcaa",
        "--trend-column-additive-analysis",
        action="store",
        required=False,
        dest="trend_column_additive_analysis",
        help="Specify the output directory where the file containing the "
        "results will be written for an additive analysis of various trend "
        "columns of the data. If this is not specified, the additive analysis "
        "won't be performed. If no directory path is specified, the file will "
        "be written to the current directory.",
    )
    parser.add_argument(
        "-tcgs",
        "--trend-column-grid-search",
        action="store",
        required=False,
        dest="trend_column_grid_search",
        help="Specify the output directory where the file containing the "
        "results will be written for a grid search of the trend columns "
        "that best improved the model from the additive analysis. "
        "If this parameter is not specified, the trend column grid search "
        "won't be performed. If no directory path is specified, the file will "
        "be written to the current directory.",
    )
    parser.add_argument(
        "-hpgs",
        "--hyperparameter-grid-search",
        action="store",
        required=False,
        dest="hyperparameter_grid_search",
        help="Specify the output directory where the file containing the "
        "results will be written for a grid search of the Random Forest "
        "Regressor hyperparameter combination that produces the best "
        "performing model. They hyperparameters searched include the "
        "number of estimators and the max depth. If this parameter is not "
        "specified, the hyperparameter grid search won't be performed. "
        "If no directory path is specified, the file will be written "
        "to the current directory.",
    )
    parser.add_argument(
        "-tgt",
        "--time-grouping-tests",
        action="store",
        required=False,
        dest="time_grouping_tests",
        help="Specify the output directory where the file containing the "
        "results will be written for each 15 minute increment time grouping "
        "using the top performing model. If this parameter is not "
        "specified, the tests won't be performed. "
        "If no directory path is specified, the file will be written "
        "to the current directory.",
    )
    parser.add_argument(
        "-bm",
        "--best-model",
        action="store_true",
        required=False,
        dest="best_model",
        help="Run the model and display the results of the best model "
        "discovered by the grid search.",
    )
    parser.add_argument(
        "-bmne",
        "--best-model-nights-excluded",
        action="store_true",
        required=False,
        dest="best_model_nights_excluded",
        help="Run the model and display the results of the best model "
        "discovered by the grid search when the nights were excluded.",
    )
    parser.add_argument(
        "-p",
        "--pickle-model",
        action="store",
        required=False,
        dest="pickle_model",
        help="When used with the --best-model option, this will save the "
        "trained model to a pickle so the trained model can be loaded for "
        "additional testing. A file path must be specified with this option "
        "for where the pickle file should be saved.",
    )
    parser.add_argument(
        "-lp",
        "--load-pickled-model",
        action="store",
        required=False,
        dest="load_pickled_model",
        help="Specify a pickle file that should be used to test the model's "
        "performance on the specified input file. When this option is specified "
        "no training will take place. All of the data will be predicted and "
        "tested to produce performance results. This is useful for testing "
        "how well the learning from one hive transfers to another.",
    )
    parser.add_argument(
        "-mcpls",
        "--model-comparison-pls",
        action="store",
        required=False,
        dest="model_comparison_pls",
        help="Trains and tests a Partial Least Squares Regression model "
        "instead of a Random Forest Regressor. This is so this model type "
        "can be compared against the Random Forest Regressor for performance."
        "Specify the output directory where the file containing the "
        "results will be written. If this parameter is not "
        "specified, this model won't be trained or tested.",
    )
    parser.add_argument(
        "-mcknn",
        "--model-comparison-knn",
        action="store",
        required=False,
        dest="model_comparison_knn",
        help="Trains and tests a K-Nearest-Neighbors Regression model "
        "instead of a Random Forest Regressor. This is so this model type "
        "can be compared against the Random Forest Regressor for performance."
        "Specify the output directory where the file containing the "
        "results will be written. If this parameter is not "
        "specified, this model won't be trained or tested.",
    )
    parser.add_argument(
        "-rfe",
        "--recursive-feature-elimination",
        action="store",
        required=False,
        dest="recursive_feature_elimination",
        help="Uses Recursive Feature Elimination to select the features to be "
        " removed. This will take all of the original data "
        "columns and all of the generated columns (such as trend) and select "
        "which features should be used. Note that the EMF will be excluded "
        "from this since the sensor wasn't calibrated for temperature. A"
        "directory must be specified with this argument to where the results"
        "file should be written.",
    )
    parser.add_argument(
        "-sfs",
        "--sequential-feature-selection",
        action="store",
        required=False,
        dest="sequential_feature_selection",
        help="Uses Sequential Feature Selection. This chooses the best feature "
        "to add based on the cross-validation score of an estimator. This "
        "will take all of the original data columns and all of the generated "
        "columns (such as trend) and select which features should be used. "
        "Note that the EMF will be excluded from this since the sensor wasn't "
        "calibrated for temperature. A directory must be specified with this "
        "argument to where the results file should be written.",
    )
    return parser.parse_args()


def delete_features(df):
    """
    Removes features from the original data that aren't
    desired to be used in the model.
    """
    if "Record Number" in df.columns:
        del df["Record Number"]
    if "Station ID" in df.columns:
        del df["Station ID"]
    if "BeeMonitorID" in df.columns:
        del df["BeeMonitorID"]
    if "Wind Direction (String)" in df.columns:
        del df["Wind Direction (String)"]
    if "UpwardMotion" in df.columns:
        del df["UpwardMotion"]
    if "DownwardMotion" in df.columns:
        del df["DownwardMotion"]
    if "LateralMotion" in df.columns:
        del df["LateralMotion"]
    # Remove the EMF variables since they were skewed by the temperature
    if "Avg. EMF (mG)" in df.columns:
        del df["Avg. EMF (mG)"]
    if "Max EMF (mG)" in df.columns:
        del df["Max EMF (mG)"]
    # Delete this one since we'll be using the solar data from USU
    if "Shortwave Radiation (W m^(-2))" in df.columns:
        del df["Shortwave Radiation (W m^(-2))"]


def normalize_column(df, column_name):
    """
    Performs a min/max normalization of a single column of a pandas dataframe.
    """
    print(f"Min/max normalizing the {column_name} column")
    minimum = np.min(df[column_name])
    maximum = np.max(df[column_name])
    norm = (df[column_name] - minimum) / (maximum - minimum)
    df[column_name] = norm


def do_feature_engineering(hive_dfs):
    """
    Perform various feature engineering techniques on the data.
    """
    # Apply the feature engineering techniques to each hive's dataframe
    for index in range(len(hive_dfs)):
        # Add a month and hour column
        util.add_col_month_and_hour(hive_dfs[index])

        # Adding these trend columns didn't appear to help as much as I hoped
        # Add a column for 2hr pressure change
        pressure_interval = timedelta(
            hours=2
        )  # 2.25 hours seems to be the best from preliminary tests
        util.add_col_trend(hive_dfs[index], "Pressure (mbar)", pressure_interval)

        # Add a column for 1hr precipitation change
        precipitation_interval = timedelta(hours=1)
        util.add_col_trend(
            hive_dfs[index], "Precipitation (Inches)", precipitation_interval
        )

        # TODO: Determine how this performs with or without the night hours
        # hive_dfs[index] = util.remove_night_hours(hive_dfs[index])


def add_trend_columns(hive_dfs, columns, specific_hours=[]):
    """
    Add trend columns for select columns at different intervals from 1 to 24 hours.
    If specific_hours is empty, all trend intervals will be generated for
    all columns in the columns list.
    If specific_hours is filled, the value at the index will be used
    as the trend interval for the column at the corresponding index
    in the columns variable.
    In this case, only one trend column will be generated for each
    specified column.
    This latter case can be for the top trend columns to be used in the
    trend column grid search.
    """
    col_names_dict = {}

    # Create a list for each column passed into the function
    # that will hold the name of each trend column generated
    # and store each list in a dictionary
    for column in columns:
        # Remove spaces since the column name will become part of the filename
        col_names_dict[column.replace(" ", "_")] = []

    for index in range(len(hive_dfs)):
        # Generate the desired trend columns for each specified column
        for interval_index, column in enumerate(columns):
            # If there weren't any specific hours specified,
            # generate a trend column for every hour from 1 to 24
            if len(specific_hours) < 1:
                for interval in range(1, 25, 1):
                    # Adjust the column name to match the dictionary field
                    col_names_dict[column.replace(" ", "_")].append(
                        util.add_col_trend(
                            hive_dfs[index], column, timedelta(hours=interval)
                        )
                    )
            # If specific trend column hours are specified, generate only one
            # trend column of the specified hour amount for each specified column.
            # The index of the hour amount specified must coincide with the index
            # of the columns passed in.
            else:
                col_names_dict[column.replace(" ", "_")].append(
                    util.add_col_trend(
                        hive_dfs[index],
                        column,
                        timedelta(hours=specific_hours[interval_index]),
                    )
                )

    # Return the dictionary containing the names of every trend column created
    return col_names_dict


def train_and_test_models(
    hive_dfs,
    hour_groupings,
    columns,
    show_plot=False,
    verbose=False,
    remove_nights=False,
    test_size=0.20,  # TrainTestSplit percentage
    random_state=1234,  # The random state so it will perform relatively the same
    n_estimators=300,  # The number of estimators to use. Set from hyperparameter grid search.
    max_depth=12,  # The max depth of each decision tree. Set from hyperparameter grid search.
    time_grouping_tests=False,  # The error term needs to be different when the samples get smaller with larger time groupings
    normalized=True,  # Indicate whether the data is normalized so rounding can be adjusted appropriately
    model_type="rfr",  # The model type to use. Supported are "rfr" (random forest regressor), "pls" (partial least squares regression", and "knn" (k-nearest-neighbors)
    knn_neighbors=16,  # The number of KNN neighbors to compare in when model_type is "knn". Default set from knn parameter search.
    pls_components=11,  # The number of components the PLS algorithm will keep. Default set from pls parameter search.
    save_pickle=None,  # The path of where to save a pickle of the trained model
    load_pickle=None,  # Where to load a pickle from. If not None, not training will take place; only testing.
):
    """
    For each hour grouping, train and test a model and produce the results.
    hive_dfs: A list of dataframes, one for each hive to be processed
    hour_groupings: A list of hour values by which the data will be grouped
                    The minimum value is the interval of the original data
    """
    for hour_grouping in hour_groupings:
        # TODO: don't keep remodifying the hive_dfs
        hive_dfs_modified = [None] * len(hive_dfs)
        for index in range(len(hive_dfs)):
            # Group the data into the specified number of hours
            hive_dfs_modified[index] = util.group_data(
                hive_dfs[index],
                time_span=timedelta(hours=hour_grouping),
                normalized=normalized,
            )

        # The input
        # Select all of the columns except the TotalMotion column
        X = hive_dfs_modified[0].loc[:, hive_dfs_modified[0].columns != "TotalMotion"]

        # The input
        # Set the inputs to only the columns specified in the parameters
        X = hive_dfs_modified[0][columns]

        # The output
        y = hive_dfs_modified[0]["TotalMotion"]

        # Get the initial train/test split objects
        # It's important that this happens before nights are removed so
        # the same data is used with and without them
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, random_state=1234
        )

        # Remove the night hours if desired
        # Do this here so the train test split will be the same with or without nights
        if remove_nights:
            X_train = util.remove_night_hours(X_train)
            X_test = util.remove_night_hours(X_test)
            y_train = util.remove_night_hours(y_train)
            y_test = util.remove_night_hours(y_test)

        # Add the data from any other files (hives) specified
        if len(hive_dfs_modified) > 1:
            for iter_index in range(len(hive_dfs_modified)):
                # Skip the first one
                index = iter_index + 1
                if index > len(hive_dfs_modified) - 1:
                    break

                y = hive_dfs_modified[index]["TotalMotion"]

                X_train_temp, X_test_temp, y_train_temp, y_test_temp = train_test_split(
                    X, y, test_size=0.20, random_state=1234
                )

                X_train = X_train.append(X_train_temp)
                y_train = y_train.append(y_train_temp)

                X_test = X_test.append(X_test_temp)
                y_test = y_test.append(y_test_temp)

        train_length = len(X_train)
        test_length = len(X_test)

        if model_type == "rfr":
            model = None
            if load_pickle:
                # Load a pretrained model from a pickle
                with open(load_pickle, "rb") as filename:
                    print(
                        f"Loading a pretrained random forest model from {load_pickle}"
                    )
                    model = pickle.load(filename)
            else:
                # Create a Random Forest Regressor
                # scikit-learn documentation indicates the following:
                # - max_features
                #   - None is empirically the best for regression.
                #   - All columns (features) will be used in every decision tree, rather than
                #     a random selection of a subset of them.
                # - boostrap
                #   - By default this is set to true. This means that the training data will
                #     be boostrapped. In other words, if there are 100 samples, a new data set
                #     will be created consisting of 100 random samples from the original data set
                #     with repeat selections allowed.
                # - n_jobs=-1 will use all cores of the processor to paralellize the operations
                model = RandomForestRegressor(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    n_jobs=-1,
                    max_features=None,
                )

                if verbose:
                    print("Fitting the model")
                model.fit(X_train, y_train)

            if verbose:
                print("Testing the model")
            y_pred = model.predict(X_test)

            column_lookup = {}
            label = 0
            for feature in X.columns:
                if feature != "TotalMotion":
                    column_lookup[label] = feature
                    label = label + 1
            feature_importances = pd.Series(model.feature_importances_).sort_values(
                ascending=False
            )
            # A list to hold both the importance value and the column name in
            # descending order. This will be returned from this function.
            feature_importances_list = []
            # Collect the importance value and the name of the column associated with it
            for key in feature_importances.keys():
                importance_string = (
                    f"{round(feature_importances[key], 5):.5f} - {column_lookup[key]}"
                )
                feature_importances_list.append(importance_string)

            pls_components = "n/a"
            knn_neighbors = "n/a"

        if model_type == "pls":
            model = None
            if load_pickle:
                # Load a pretrained model from a pickle
                with open(load_pickle, "rb") as filename:
                    print(f"Loading a pretrained PLS model from {load_pickle}")
                    model = pickle.load(filename)
            else:
                model = PLSRegression(n_components=pls_components)
                model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_pred = y_pred.reshape(1, -1)[0]  # Convert to a single list of values
            feature_importances_list = "n/a"
            n_estimators = "n/a"
            max_depth = "n/a"
            knn_neighbors = "n/a"

        if model_type == "knn":
            model = None
            if load_pickle:
                # Load a pretrained model from a pickle
                with open(load_pickle, "rb") as filename:
                    print(f"Loading a pretrained KNN model from {load_pickle}")
                    model = pickle.load(filename)
            else:
                model = KNeighborsRegressor(
                    n_neighbors=knn_neighbors, weights="distance", n_jobs=-1
                )
                model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            feature_importances_list = "n/a"
            n_estimators = "n/a"
            max_depth = "n/a"
            pls_components = "n/a"

        if save_pickle:
            # Write the trained model to a file so it can be loaded later
            with open(save_pickle, "wb") as filename:
                print(f"Saving the trained model to a pickle at {save_pickle}")
                pickle.dump(model, filename)

        r2_score = metrics.r2_score(y_test, y_pred)

        if show_plot:
            # Plot the predicted vs the actual
            comparison_df = y_test.to_frame()
            comparison_df.columns = ["TotalMotion"]
            comparison_df["TotalMotionPredicted"] = y_pred
            comparison_df["TotalMotion"].plot(legend="True")
            comparison_df["TotalMotionPredicted"].plot(style="--", legend="True")

            plt.title("RandomForest")
            plt.xlabel("Time")
            plt.ylabel("TotalMotion")
            plt.show()
            # Uncomment to block between each plot
            # plt.show(block=False)

        # Calculate the residuals (the difference between the actual and predicted)
        residuals = y_test - y_pred

        # Calculate the 95% confidence interval of the errors
        # https://www.statology.org/confidence-intervals-python/
        confidence_interval = stats.norm.interval(
            alpha=0.95, loc=np.mean(residuals), scale=stats.sem(residuals)
        )
        confidence_interval_span = confidence_interval[1] - confidence_interval[0]

        ## Uncomment for additional plots
        #
        # Plot the residuals in ascending order
        # if show_plot:
        #    plt.plot(range(len(residuals)), residuals.sort_values(ascending=True))
        #    plt.show()
        ## Display a Normal Q-Q plot of the residuals to see if they are
        ## normally distributed
        # fig = sm.qqplot(residuals)
        # plt.title("Normal Q-Q Plot")
        # plt.show()

        ## Plot the predicted - actual errors squared sorted from least to greatest
        # plt.figure()
        # plt.title("Squared Errors")
        # plt.xlabel("Index")
        # plt.ylabel("Error Squared")
        # errors = []
        # for index in range(len(y_test)):
        #    error_squared = (y_pred[index] - y_test[index])**2
        #    errors.append(error_squared)
        # errors.sort()
        # plt.scatter(range(len(errors)), errors)
        # plt.show()

        # Display the confusion matrix
        # plt.figure(2)
        # confusion_matrix = pd.DataFrame(metrics.confusion_matrix(y_test, y_pred))
        # sn.heatmap(confusion_matrix, annot=True)
        # plt.show()

        # Calculate the corrected Akaike Information Criteria (AICc) for the model
        if time_grouping_tests:
            error_term = 2.1
        else:
            # The error term needs to be larger when the time groupings get larger
            error_term = 2
        print(f"error_term: {error_term}")
        n = test_length
        rss = np.sum(np.square(residuals))
        K = 2 * len(columns) + error_term
        print(f"n: {n}")
        print(f"K: {K}")
        print(f"rss: {rss}")
        aicc = n * math.log(rss / n) + 2 * K + ((2 * K * (K + 1)) / (n - K - 1))

        # Clear out the modified hive data frames
        hive_dfs_modified = []

        if verbose:
            print("")
            print(f"r2_score: {r2_score}")
            print(f"aicc: {aicc}")
            print(f"confidence_interval: {confidence_interval}")
            print(f"confidence_interval_span: {confidence_interval_span}")
            print(f"columns: {columns}")
            print(f"feature_importances_list: {feature_importances_list}")
            print(f"time_periodicity_hrs: {hour_groupings}")
            print(f"test_size: {test_size}")
            print(f"random_state: {random_state}")
            print(f"train_length: {train_length}")
            print(f"test_length: {test_length}")
            print(f"n_estimators: {n_estimators}")
            print(f"max_depth: {max_depth}")
            print(f"knn_neighbors: {knn_neighbors}")
            print(f"pls_components: {pls_components}")
            print("")

        # Return information about the model and the results
        return (
            r2_score,
            aicc,
            confidence_interval,
            confidence_interval_span,
            columns,
            feature_importances_list,
            hour_groupings,
            test_size,
            random_state,
            train_length,
            test_length,
            n_estimators,
            max_depth,
            knn_neighbors,
            pls_components,
        )


def original_column_grid_search(
    hive_dfs,
    input_filename,
    output_directory,
    output_columns,
    remove_nights=False,
    normalized=True,
):
    """
    This function performs a grid search on a list of columns.
    This excludes the frequencies and peaks of the EMR data.
    It also excludes the EMF variables since the sensor used was not
    calibrated for temperature fluctuations.
    It does not include the trend columns, since those will be added later.
    If it takes 1 second per model trained, for 14 variables, it should take
    approximately 4.55 hours to complete the grid search. If I used all of the
    original 28 columns (excluding EMF), it would take approximately 8.5 years
    to complete the grid search. This would require a different feature
    elimination or selection method to be employed.
    """
    # The list of all columns to search in different combinations
    columns_to_use = [
        "Temperature (F)",
        "Pressure (mbar)",
        "Humidity (%)",
        "Wind Direction (Degrees)",
        "Wind Speed (MPH)",
        "Wind Gust (MPH)",
        "Precipitation (Inches)",
        "Avg. RF Watts (W)",
        "Avg. RF Density (W m^(-2))",
        "Avg. Total Density (W m^(-2))",
        "Avg. EF (V/m)",
        "Shortwave Radiation USU (MJ/m^2)",
        "Month",
        "Hour",
    ]

    # Calculate the total number of iterations that will be performed
    # so progress can be reported while searching
    search_progress = 0
    total_iterations = 0
    for i in range(len(columns_to_use)):
        comb = combinations(columns_to_use, i + 1)
        num_items = len(list(comb))
        total_iterations = total_iterations + num_items

    # Perform the grid search at 15 minutes time intervals.
    # This is the original data interval.
    grid_search_time_grouping = [0.25]

    # List to hold the results of each model test
    results_list = []

    # The start time of the grid search
    start_time_raw = datetime.now()
    start_time = start_time_raw.strftime("%Y-%m-%d_%H:%M:%S")
    print(f"Start Time: {start_time}")

    # Train a model for every combination of the columns_to_use
    for num_cols_in_combo in range(len(columns_to_use)):
        # Generate a combination
        combo = combinations(columns_to_use, num_cols_in_combo + 1)
        combo_list = list(combo)

        # Train a model for each combination
        for test_column_combo in combo_list:
            if not remove_nights:
                results = train_and_test_models(
                    hive_dfs,
                    [0.25],
                    list(test_column_combo),
                    n_estimators=100,
                    max_depth=6,
                    verbose=True,
                    normalized=normalized,
                )
            else:
                results = train_and_test_models(
                    hive_dfs,
                    [0.25],
                    list(test_column_combo),
                    n_estimators=100,
                    max_depth=6,
                    verbose=True,
                    remove_nights=True,
                    normalized=normalized,
                )
            results_list.append(results)

            search_progress = search_progress + 1
            print(f"Progress: {search_progress}/{total_iterations}")

        # Create a pandas dataframe from the results list
        grid_search_df = pd.DataFrame(results_list, columns=output_columns)

        # Write the results to a file after each num_cols_in_combo iteration.
        # This is a saftey measure so progress can be somewhat resumed if
        # the search gets interrupted at some point.
        input_filename_sanitized = os.path.splitext(os.path.basename(input_filename))[0]

        if not remove_nights:
            grid_search_df.to_csv(
                path_or_buf=f"{output_directory}/rfr_grid_search_on_{input_filename_sanitized}__{start_time}.csv",
                index=False,
            )
        else:
            grid_search_df.to_csv(
                path_or_buf=f"{output_directory}/rfr_grid_search_on_{input_filename_sanitized}_nights_removed__{start_time}.csv",
                index=False,
            )

    end_time_raw = datetime.now()
    end_time = end_time_raw.strftime("%Y-%m-%d_%H:%M:%S")
    print(f"End Time: {end_time}")
    print(f"Total Search Time: {end_time_raw - start_time_raw}")


def trend_column_additive_analysis(
    hive_dfs,
    input_filename,
    output_directory,
    trend_column_names,
    output_columns,
    normalized=True,
):
    """
    Loops through every trend column and trains a model with the
    trend column added to the grid-search-selected best model.
    The test results for all columns are then written to a file.
    """
    # The list of base columns to search in different combinations
    # These came from the best performing model as found by the
    # column grid search.
    # Note: These must be manually updated after running the original
    # column grid search and the top columns have been selected.
    columns_to_use = [
        "Temperature (F)",
        "Pressure (mbar)",
        "Humidity (%)",
        "Wind Speed (MPH)",
        "Shortwave Radiation USU (MJ/m^2)",
        "Month",
        "Hour",
    ]

    # Calculate the total number of iterations that will be performed
    # so progress can be reported while searching
    analysis_progress = 0
    total_iterations = len(trend_column_names) * len(
        trend_column_names[list(trend_column_names.keys())[0]]
    )

    # Perform the tests at 15 minutes time intervals.
    # This is the original data interval.
    time_grouping = [0.25]

    # List to hold the results of each model test
    results_list = []

    # The start time of the additive tests
    start_time_raw = datetime.now()
    start_time = start_time_raw.strftime("%Y-%m-%d_%H:%M:%S")
    print(f"Start Time: {start_time}")

    # Train a model for each trend column individually added to the base
    # best columns to use
    print(f"trend_column_names.keys(): {trend_column_names.keys()}")
    for key in trend_column_names.keys():
        print("New Iteration...")
        for trend_col in trend_column_names[key]:
            print(f"trend_col: {trend_col}")
            results = train_and_test_models(
                hive_dfs,
                time_grouping,
                columns_to_use + [trend_col],
                verbose=True,
                normalized=normalized,
            )
            results_list.append(results)

            analysis_progress = analysis_progress + 1
            print(f"Progress: {analysis_progress}/{total_iterations}")

        # Create a pandas dataframe from the results list
        analysis_df = pd.DataFrame(results_list, columns=output_columns)

        # Write the results to a file
        # Note that this is not part of the loop above like in the base column
        # grid search. This is because this will have far fewer iterations.
        input_filename_sanitized = os.path.splitext(os.path.basename(input_filename))[0]

        # Write the results to a file after each num_cols_in_combo iteration.
        # This is a safety measure so progress can be somewhat resumed if
        # the search gets interrupted at some point.
        input_filename_sanitized = os.path.splitext(os.path.basename(input_filename))[0]

        # Sanitize the key so it doesn't behave as a directory when writing the file
        key_sanitized = key.replace("/", "")
        key_sanitized = key_sanitized.replace("(", "")
        key_sanitized = key_sanitized.replace(")", "")
        key_sanitized = key_sanitized.replace("^", "")

        analysis_df.to_csv(
            path_or_buf=f"{output_directory}/{key_sanitized}_rfr_trend_col_analysis_{input_filename_sanitized}__{start_time}.csv",
            index=False,
        )

        # Clear out the results for the next trend columns
        results_list = []

    end_time_raw = datetime.now()
    end_time = end_time_raw.strftime("%Y-%m-%d_%H:%M:%S")
    print(f"End Time: {end_time}")
    print(f"Total Additive Analysis Time: {end_time_raw - start_time_raw}")


def trend_column_grid_search(
    hive_dfs,
    input_filename,
    output_directory,
    trend_column_names,
    columns_to_use,
    output_columns,
    normalized=True,
):
    """
    This function performs a grid search on the top performing trend columns.
    The 7 top trend columns and associated intervals will be added to the
    base grid-search-selected columns in all possible combinations. Only
    the trend columns will be in different combinations, the base columns
    will not change. The test results will be written to a file so the top
    model can be selected. This will determine whether it's worth it to add
    any of the trend columns, or if the base columns are sufficient.
    """
    # Calculate the total number of iterations that will be performed
    # so progress can be reported while searching
    search_progress = 0
    total_iterations = 0
    for i in range(len(trend_column_names)):
        comb = combinations(trend_column_names, i + 1)
        num_items = len(list(comb))
        total_iterations = total_iterations + num_items

    # Perform the grid search at 15 minutes time intervals.
    # This is the original data interval.
    grid_search_time_grouping = [0.25]

    # List to hold the results of each model test
    results_list = []

    # The start time of the grid search
    start_time_raw = datetime.now()
    start_time = start_time_raw.strftime("%Y-%m-%d_%H:%M:%S")
    print(f"Start Time: {start_time}")

    # Train a model for every combination of the trend columns along with the unchanged base columns
    for num_cols_in_combo in range(len(trend_column_names)):
        # Generate a combination
        combo = combinations(trend_column_names, num_cols_in_combo + 1)
        combo_list = list(combo)

        # Train a model for each combination
        for test_column_combo in combo_list:
            results = train_and_test_models(
                hive_dfs,
                grid_search_time_grouping,
                # Use the base columns (unchanged) with each trend column combo
                columns_to_use + list(test_column_combo),
                verbose=True,
                normalized=normalized,
            )
            results_list.append(results)

            search_progress = search_progress + 1
            print(f"Progress: {search_progress}/{total_iterations}")

        # Create a pandas dataframe from the results list
        grid_search_df = pd.DataFrame(results_list, columns=output_columns)

        # Write the results to a file after each num_cols_in_combo iteration.
        # This is a saftey measure so progress can be somewhat resumed if
        # the search gets interrupted at some point.
        input_filename_sanitized = os.path.splitext(os.path.basename(input_filename))[0]

        grid_search_df.to_csv(
            path_or_buf=f"{output_directory}/trend_column_rfr_grid_search_on_{input_filename_sanitized}__{start_time}.csv",
            index=False,
        )

    end_time_raw = datetime.now()
    end_time = end_time_raw.strftime("%Y-%m-%d_%H:%M:%S")
    print(f"End Time: {end_time}")
    print(f"Total Search Time: {end_time_raw - start_time_raw}")


def hyperparameter_grid_search(
    hive_dfs,
    input_filename,
    output_directory,
    columns_to_use,
    output_columns,
    normalized=True,
):
    """
    Perform a grid search of the Random Forest Regressor hyperparameter
    combination that produces the best performing model using the base
    columns and trend columns selected from previous grid searches.
    The hyperparameters searched include the number of estimators
    and the max depth. If this parameter is not specified, the
    hyperparameter grid search won't be performed. If no directory path
    is specified, the file will be written to the current directory.",
    The columns_to_use has the ultimate top base columns and
    top trend columns as previously selected.
    """
    # The values to grid search
    num_estimators_values = list(range(100, 501, 25))
    max_depth_values = list(range(2, 21, 1))

    # Calculate the total number of iterations that will be performed
    # so progress can be reported while searching
    search_progress = 0
    total_iterations = len(num_estimators_values) * len(max_depth_values)

    # Perform the grid search at 15 minutes time intervals.
    # This is the original data interval.
    grid_search_time_grouping = [0.25]

    # List to hold the results of each model test
    results_list = []

    # The start time of the grid search
    start_time_raw = datetime.now()
    start_time = start_time_raw.strftime("%Y-%m-%d_%H:%M:%S")
    print(f"Start Time: {start_time}")

    # Train a model for every combination of the number of estimators
    # and the max depth values
    for num_estimators_value in num_estimators_values:
        for max_depth_value in max_depth_values:
            results = train_and_test_models(
                hive_dfs,
                grid_search_time_grouping,
                columns_to_use,
                verbose=True,
                n_estimators=num_estimators_value,
                max_depth=max_depth_value,
            )
            results_list.append(results)

            search_progress = search_progress + 1
            print(f"Progress: {search_progress}/{total_iterations}")

        # Create a pandas dataframe from the results list
        grid_search_df = pd.DataFrame(results_list, columns=output_columns)

        # Write the results to a file after each num_cols_in_combo iteration.
        # This is a saftey measure so progress can be somewhat resumed if
        # the search gets interrupted at some point.
        input_filename_sanitized = os.path.splitext(os.path.basename(input_filename))[0]

        grid_search_df.to_csv(
            path_or_buf=f"{output_directory}/hyperparameter_rfr_grid_search_on_{input_filename_sanitized}__{start_time}.csv",
            index=False,
        )

    end_time_raw = datetime.now()
    end_time = end_time_raw.strftime("%Y-%m-%d_%H:%M:%S")
    print(f"End Time: {end_time}")
    print(f"Total Search Time: {end_time_raw - start_time_raw}")


def time_grouping_tests(
    hive_dfs,
    input_filename,
    output_directory,
    columns_to_use,
    output_columns,
    normalized=True,
):
    """
    Using the best obtained model from the base column
    grid search, trend column grid search, and
    hyperparameter grid search, produce the test results
    for time groupings from 15 minutes to 24 hours
    in 15 minute time increments.
    The columns_to_use has the ultimate top base columns and
    top trend columns as previously selected.
    """
    # Calculate the total number of iterations that will be performed
    # so progress can be reported while searching
    search_progress = 0
    total_iterations = 24 * 4

    # Perform the tests at 15 minute time increments up through 24 hours
    time_groupings = np.arange(0.25, 24.1, 0.25)

    # List to hold the results of each model test
    results_list = []

    # The start time of the grid search
    start_time_raw = datetime.now()
    start_time = start_time_raw.strftime("%Y-%m-%d_%H:%M:%S")
    print(f"Start Time: {start_time}")

    # Train a model for each different time grouping
    for time_grouping in time_groupings:
        results = train_and_test_models(
            hive_dfs,
            [time_grouping],
            columns_to_use,
            verbose=True,
            # Use the defaults for the hyperparameters, since they have
            # been set to the best performing values.
            time_grouping_tests=True,
            # Do Not Uncomment the model_type here unless it is explicitly
            # intended that a certain model_type is tested instead of the random forest.
            # model_type="pls",  # "knn",
            # knn_neighbors=16,
            # pls_components=11,
        )
        results_list.append(results)

        search_progress = search_progress + 1
        print(f"Progress: {search_progress}/{total_iterations}")

        # Create a pandas dataframe from the results list
        results_df = pd.DataFrame(results_list, columns=output_columns)

        # Write the results to a file after each num_cols_in_combo iteration.
        # This is a saftey measure so progress can be somewhat resumed if
        # the search gets interrupted at some point.
        input_filename_sanitized = os.path.splitext(os.path.basename(input_filename))[0]

        results_df.to_csv(
            path_or_buf=f"{output_directory}/time_groupings_rfr_test_results_{input_filename_sanitized}__{start_time}.csv",
            index=False,
        )

    end_time_raw = datetime.now()
    end_time = end_time_raw.strftime("%Y-%m-%d_%H:%M:%S")
    print(f"End Time: {end_time}")
    print(f"Total Test Time: {end_time_raw - start_time_raw}")


def run_best_model(
    hive_dfs,
    columns_to_use,
    columns_to_use_nights_excluded=[],
    nights_excluded=False,
    normalized=True,
    save_pickle=None,
    load_pickle=None,
):
    """
    Runs the best model as obtained from a previous grid search.
    """
    if len(columns_to_use_nights_excluded) < 1:
        columns_to_use_nights_excluded = columns_to_use

    # 15 minutes time intervals.
    # This is the original data interval.
    time_grouping = [0.25]
    # For testing purposes: This can be set to different hours to get a score
    #time_grouping = [1.0]

    # Train the model. This will print the results to the console.
    # The default hyperparameters will be used in training the models since
    # these have been set to the best parameters found.
    if nights_excluded:
        train_and_test_models(
            hive_dfs,
            time_grouping,
            columns_to_use_nights_excluded,
            show_plot=True,
            verbose=True,
            remove_nights=True,
            normalized=normalized,
            save_pickle=save_pickle,
            load_pickle=load_pickle,
        )
    else:
        train_and_test_models(
            hive_dfs,
            time_grouping,
            # Keep this uncommented for general use
            columns_to_use,
            # Left for testing to replace columns_to_use if desired
            # The following can be uncommented one at a time to see the various
            # contributions of different combintations of the top predictor variables.
            # ["Humidity (%) 9:00:00 Trend"],
            # ["Temperature (F)"],
            # ["Shortwave Radiation USU (MJ/m^2)"],
            # ["Humidity (%) 9:00:00 Trend","Shortwave Radiation USU (MJ/m^2)"],
            # ["Humidity (%) 9:00:00 Trend","Temperature (F)"],
            # ["Temperature (F)", "Shortwave Radiation USU (MJ/m^2)"],
            # ["Humidity (%) 9:00:00 Trend","Temperature (F)", "Shortwave Radiation USU (MJ/m^2)"],
            # Left for testing to replace columns_to_use if desired
            # These test the contributions of certain variables, as well as assesses the
            # predictive ability of using EMR variables alone.
            # ["Avg. EF (V/m)"],
            # ["Avg. RF Watts (W)"],
            # ["Pressure (mbar)"],
            # ["Avg. Total Density (W m^(-2))"],
            # ["Wind Speed (MPH)"],
            # ["Avg. RF Watts (W)", "Avg. RF Density (W m^(-2))", "Avg. Total Density (W m^(-2))", "Avg. EF (V/m)"],
            # ["Avg. RF Density (W m^(-2))", "Avg. Total Density (W m^(-2))", "Avg. EF (V/m)"],
            # ["Avg. Total Density (W m^(-2))", "Avg. EF (V/m)"],
            show_plot=True,
            verbose=True,
            normalized=normalized,
            save_pickle=save_pickle,
            load_pickle=load_pickle,
        )


def model_for_comparison(
    hive_dfs,
    input_filename,
    output_directory,
    columns_to_use,
    output_columns,
    model_type,
    normalized=True,
):
    """
    Using the best obtained model from the base column
    grid search, trend column grid search, and
    hyperparameter grid search, train and test a Partial Least
    Squares Regression model or K-Nearest-Neighbors model.
    The results will be written to a file so they can be compared
    to the results of the random forest regressor.
    """
    print(
        f"Training and testing a {model_type} model for comparison against the Random forest Regressor model."
    )
    # 15 minutes time intervals.
    # This is the original data interval.
    time_grouping = [0.25]

    # List to hold the results of each model test
    results_list = []

    # The start time of the grid search
    start_time_raw = datetime.now()
    start_time = start_time_raw.strftime("%Y-%m-%d_%H:%M:%S")
    print(f"Start Time: {start_time}")
    results_list = []

    if model_type == "pls":
        # Train a model for several different components to keep in the PLS algorithm
        component_values = range(
            1, 12, 1
        )  # 1 to 11 (the number of features). This adheres to scikit-learn's guideline of being within [1, min(n_samples, n_features, n_targets)]
        for n_components in component_values:
            results = train_and_test_models(
                hive_dfs,
                time_grouping,
                columns_to_use,
                verbose=False,
                normalized=normalized,
                # Use the defaults for the hyperparameters, since they have
                # been set to the best performing values.
                model_type=model_type,
                pls_components=n_components,
            )
            results_list.append(results)
    if model_type == "knn":
        # Train a model for several different k values (number of neighbors)
        k_values = range(1, 21, 1)  # 1 to 20 in increments of 1
        for k in k_values:
            results = train_and_test_models(
                hive_dfs,
                time_grouping,
                columns_to_use,
                verbose=False,
                normalized=normalized,
                # Use the defaults for the hyperparameters, since they have
                # been set to the best performing values.
                model_type=model_type,
                knn_neighbors=k,
            )
            results_list.append(results)

    # Create a pandas dataframe from the results list
    results_df = pd.DataFrame(results_list, columns=output_columns)

    # Write the results to a file after each num_cols_in_combo iteration.
    input_filename_sanitized = os.path.splitext(os.path.basename(input_filename))[0]

    results_df.to_csv(
        path_or_buf=f"{output_directory}/{model_type}_test_results_{input_filename_sanitized}__{start_time}.csv",
        index=False,
    )

    end_time_raw = datetime.now()
    end_time = end_time_raw.strftime("%Y-%m-%d_%H:%M:%S")
    print(f"End Time: {end_time}")
    print(f"Total Test Time: {end_time_raw - start_time_raw}")


def perform_feature_selection(
    hive_dfs,
    output_directory,
    input_filename,
    output_columns,
    method="rfe", # This is the default because it's faster
    time_groupings=[0.25],
    normalized=True,
):
    """
    Perform Recursive Feature Elimination
    or Sequential Feature Selection to obtain the best number of features,
    and which features produce the best model.
    The value of the "method" parameter can be either "rfe" or "sfs" for
    Recursive Feature Elimination or Sequential Feature Selection respectively.
    """
    # The input
    # Select all of the columns except the TotalMotion column since that's what's
    # being predicted
    X = hive_dfs[0].loc[:, hive_dfs[0].columns != "TotalMotion"]

    # IMPORTANT: Only uncomment this when certain columns should be used
    # columns = ["Temperature (F)", "Humidity (%)"]#, "Wind Speed (MPH)"]
    columns = X.columns

    # Only keep the specified columns in the 'columns' variable
    X = hive_dfs[0][columns]

    print("\nStarting Columns:\n")
    for column in X.columns:
        print(f"\t{column}")
    print("")

    # The output
    y = hive_dfs[0]["TotalMotion"]

    # Get the initial train/test split objects
    # It's important that this happens before nights are removed so
    # the same data is used with and without them
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=1234
    )

    estimator = RandomForestRegressor(
        n_estimators=300,  # Set to the same as the default for train_and_test_models
        max_depth=12,  # Set to the same as the default for train_and_test_models
        n_jobs=-1,  # Parallelize training
        max_features=None,
    )

    selector = None
    if method == "rfe":
        # After running RFECV, it selected 50-100 different features.
        # For consistentcy with the other tests, I think I'll use regular RFE
        # and only select a maximum of 11 features. Then I can better compare
        # the selection methods.
        # Create the Recursive Feature Elimination with Cross-validation selector
        # With step = 1, one feature will be eliminated at a time
        # cv = 5 is the default for the cross-validation splitting strategy. Specifies number of folds.
        # NOTE: This is left here for testing purposes if desired.
        # selector = RFECV(
        #     estimator, step=0.05, cv=5, n_jobs=-1, min_features_to_select=11
        # )
        
        # Use RFE to select the top 11 features while removing 5% of the features
        # after each iteration.
        selector = RFE(
            estimator, step=0.05, n_features_to_select=11
        )
    elif method == "sfs":
        # Create a Sequential Feature Selector
        # By default this will use 5-fold cross-validation
        selector = SequentialFeatureSelector(estimator, n_features_to_select=11, direction='forward')

    # Begin timing the selection process
    start_time_raw = datetime.now()
    start_time = start_time_raw.strftime("%Y-%m-%d_%H:%M:%S")
    print(f"Start Time: {start_time}")

    # Perform the feature elimination/selection
    # Only use the training data. 
    # We only want to use the test data for final testing.
    selector.fit(X_train, y_train)

    end_time_raw = datetime.now()
    end_time = end_time_raw.strftime("%Y-%m-%d_%H:%M:%S")
    print(f"End Time: {end_time}")
    print(f"Total Test Time: {end_time_raw - start_time_raw}")
    print("")

    # Print the selection in a more human-readable format and
    # create a list of selected columns
    columns_to_use = []
    for index, column in enumerate(X.columns):
        if method == "rfe":
            print(f"{selector.support_[index]}, {selector.ranking_[index]} : {column}")
        elif method == "sfs":
            print(f"{selector.get_support()[index]} : {column}")

        if selector.support_[index]:
            columns_to_use.append(column)

    print("\nTraining and testing a model with the selected features")

    # Call train_and_test and pass it the columns selected to see the
    # performance of a model with the selected features
    results = train_and_test_models(
        hive_dfs,
        time_groupings,
        columns_to_use,
        show_plot=False,
        verbose=True,
        normalized=normalized,
    )
    # Create a pandas dataframe from the results list
    results_df = pd.DataFrame([results], columns=output_columns)

    # Write the results to a file
    input_filename_sanitized = os.path.splitext(os.path.basename(input_filename))[0]
    results_df.to_csv(
        path_or_buf=f"{output_directory}/rfr_{method}_selected_model_results_{input_filename_sanitized}__{start_time}.csv",
        index=False,
    )


def main():
    """
    The main driver for the program.
    """
    args = get_args()

    # List to hold the dataframe for each file loaded
    hive_dfs = []

    # Handle when a single file is specified
    if args.input_file:
        df = pd.read_csv(
            args.input_file,
            header=0,
            index_col=["Time"],
            parse_dates=True,
            infer_datetime_format=True,
        )
        hive_dfs.append(df)
    # Handle when a file containing a list of files is specified
    elif args.input_file_list:
        with open(args.input_file_list) as file:
            # Create a dataframe for each file specified
            for file_path in file:
                # Ignore lines that are commented out
                if (
                    file_path[0] != "#"
                    and file_path[0] != "\n"
                    and file_path[0] != ""
                    and file_path[0] != " "
                ):
                    file_path = file_path.replace("\n", "")
                    print(f"Using data file: {file_path}")
                    hive_dfs.append(
                        pd.read_csv(
                            file_path,
                            header=0,
                            index_col=["Time"],
                            parse_dates=True,
                            infer_datetime_format=True,
                        )
                    )
    else:
        print("ERROR: No input file specified.")
        sys.exit(1)

    # Shorten the data to the time range specified.
    temp = []
    if args.time_range is not None:
        for index, df in enumerate(hive_dfs):
            # Parse the start and end time
            start_end_times = args.time_range.split(",")
            # Grab only the data within the time range
            # hive_dfs[index] = hive_dfs[index].loc[start_end_times[0] : start_end_times[1]]
            temp.append(hive_dfs[index].loc[start_end_times[0] : start_end_times[1]])
        # This is to avoid the pandas set on a copy of a slice warning
        hive_dfs = copy.deepcopy(temp)
        temp.clear()

    # Indicate whether the data is normalized or not
    normalized = False

    print("Replacing NaNs in the 'TotalMotion' column with '0'")
    print("Linearly interpolating NaNs in the data")
    print("Removing data gaps")
    print("Setting remaining NaNs to 0")
    print("Normalizing the data")
    print("Adding the Month and Hour columns")
    for index in range(len(hive_dfs)):
        # Remove features that aren't needed
        delete_features(hive_dfs[index])

        # Fill the night hours with 0 instead of NaN
        hive_dfs[index]["TotalMotion"].replace(math.nan, 0, inplace=True)

        # Linearly interpolate up to 4 consecutive NaNs
        hive_dfs[index] = hive_dfs[index].interpolate(limit=4)

        # Delete any remaining rows with non-interpolatable NaNs with at least
        # 10 missing column values.
        # This will get rid of large data gaps
        hive_dfs[index] = hive_dfs[index].dropna(axis="index", thresh=10)

        # Fill remaining NaNs with 0 (this will probably be in the USU solar column)
        hive_dfs[index] = hive_dfs[index].fillna(0)

        # Note: This commented out section may be obsolete as of 7/24/2021
        # Normalize the TotalMotion column accross all hives
        # Note: If enabling the multiple hive tests, be sure to uncomment this.
        # normalize_column(hive_dfs[index], "TotalMotion")

        # Normalize all columns by doing a min/max scaling operation
        hive_dfs[index] = (hive_dfs[index] - hive_dfs[index].min()) / (
            hive_dfs[index].max() - hive_dfs[index].min()
        )
        normalized = True

        # Add the month and hour columns
        util.add_col_month_and_hour(hive_dfs[index])

    # The grid search output columns
    grid_search_output_columns = (
        "r^2_score",
        "aicc",
        "95%_confidence_interval",
        "95%_confidence_interval_span",
        "input_columns",
        "feature_importances",
        "data_periodicity_hrs",
        "train_test_split__test_size",
        "train_test_split__random_state",
        "train_length",
        "test_length",
        "n_estimators",
        "max_depth",
        "knn_neighbors",
        "pls_components",
    )

    # Grid search the original columns to determine the best ones to use
    # This will be the base model for the future model improvements
    if args.orig_col_grid_search:
        original_column_grid_search(
            hive_dfs,
            args.input_file,
            args.orig_col_grid_search,
            grid_search_output_columns,
            normalized=normalized,
        )
        print("Done performing the original column grid search. Returning.")
        return

    # Note: This variable must be populated with the top performing base
    # columns as selected by the model_selection.py script on the output of
    # the original_column_grid_search.
    base_columns = [
        "Temperature (F)",
        "Pressure (mbar)",
        "Humidity (%)",
        "Wind Speed (MPH)",
        "Shortwave Radiation USU (MJ/m^2)",
        "Month",
        "Hour",
    ]

    # Note: This variable must be populated with the top selected trend
    # columns as selected by the model_selection.py script from each
    # additive analysis resultant file.
    top_trend_columns = [
        "Temperature (F)",
        "Pressure (mbar)",
        "Humidity (%)",
        "Wind Speed (MPH)",
        "Shortwave Radiation USU (MJ/m^2)",
        # Omitting the Month since it doesn't make sense to have its trend included
        # "Month",
        # Omitting the Hour since it doesn't make sense to have its trend included
        # "Hour",
    ]

    # Note: This variable must be populated with the trend intervals
    # associated with each of the top trend columns selected by the
    # trend column grid search. The index must correspond to the
    # associated column in the top_trend_columns variable.
    top_trend_column_intervals = [8, 11, 9, 8, 5]  # 24, 5]

    if args.trend_column_additive_analysis:
        # This is a slow operation
        # Generate trend columns for each of the top base columns
        trend_column_names_dict = add_trend_columns(hive_dfs, base_columns)
    # Don't add the trend columns when doing feature selection.
    # The columns will be added later for those tests.
    elif (
        not args.recursive_feature_elimination and not args.sequential_feature_selection
    ):
        # Only generate the selected top trend columns
        trend_column_names_dict = add_trend_columns(
            hive_dfs, top_trend_columns, top_trend_column_intervals
        )

    # Grid search the original columns with the night hours removed to determine
    # the best ones to use
    if args.orig_col_grid_search_nights_excluded:
        original_column_grid_search(
            hive_dfs,
            args.input_file,
            args.orig_col_grid_search_nights_excluded,
            grid_search_output_columns,
            remove_nights=True,
            normalized=normalized,
        )

    # Determine which trend columns and intervals improve the base model
    # by adding each trend column individually to the other base columns
    # and measuring the performance of the model for comparisons.
    if args.trend_column_additive_analysis:
        trend_column_additive_analysis(
            hive_dfs,
            args.input_file,
            args.trend_column_additive_analysis,
            trend_column_names_dict,
            grid_search_output_columns,
            normalized=normalized,
        )

    # Grid search the best trend intervals found from the additive analysis
    # along with the grid-search-selected best original columns
    if args.trend_column_grid_search:
        # Put all of the trend column names into a single list
        trend_column_names = []
        for key in trend_column_names_dict:
            trend_column_names = trend_column_names + trend_column_names_dict[key]

        trend_column_grid_search(
            hive_dfs,
            args.input_file,
            args.trend_column_grid_search,
            trend_column_names,
            base_columns,
            grid_search_output_columns,
            normalized=normalized,
        )

    # The ultimate top base columns and top grid-searched trend columns to use
    # Note: These columns need to be manually updated after running the previous
    # tests and obtaining the best models using the model_selection.py script.
    top_columns_to_use = [
        "Temperature (F)",
        "Pressure (mbar)",
        "Humidity (%)",
        "Wind Speed (MPH)",
        "Shortwave Radiation USU (MJ/m^2)",
        "Month",
        "Hour",
        "Temperature (F) 8:00:00 Trend",
        "Humidity (%) 9:00:00 Trend",
        "Wind Speed (MPH) 8:00:00 Trend",
        "Shortwave Radiation USU (MJ/m^2) 5:00:00 Trend",
    ]

    # Grid search the best Random Forest Regressor hyperparameters to use
    if args.hyperparameter_grid_search:
        hyperparameter_grid_search(
            hive_dfs,
            args.input_file,
            args.hyperparameter_grid_search,
            top_columns_to_use,
            grid_search_output_columns,
            normalized=normalized,
        )

    # Obtain accuracy results for different time groupings of the data
    if args.time_grouping_tests:
        time_grouping_tests(
            hive_dfs,
            args.input_file,
            args.time_grouping_tests,
            top_columns_to_use,
            grid_search_output_columns,
            normalized=normalized,
        )

    # Train and produce the results of the best performing model discovered by
    # the grid search
    if args.best_model and args.load_pickled_model is None:
        run_best_model(
            hive_dfs,
            top_columns_to_use,
            grid_search_output_columns,
            normalized=normalized,
            save_pickle=args.pickle_model,
        )

    # Train and produce the results of the best performing model discovered by
    # the grid search when the nights were removed
    if args.best_model_nights_excluded and args.load_pickled_model is None:
        run_best_model(
            hive_dfs,
            top_columns_to_use,
            grid_search_output_columns,
            nights_excluded=True,
            normalized=normalized,
        )

    # Train and test a Partial Least Square Regression model
    if args.model_comparison_pls:
        print(args.model_comparison_pls)
        model_for_comparison(
            hive_dfs,
            args.input_file,
            args.model_comparison_pls,  # Output directory
            top_columns_to_use,
            grid_search_output_columns,
            model_type="pls",
            normalized=normalized,
        )

    # Train and test a K-Nearest-Neighbors Regression model
    if args.model_comparison_knn:
        model_for_comparison(
            hive_dfs,
            args.input_file,
            args.model_comparison_knn,  # Output directory
            top_columns_to_use,
            grid_search_output_columns,
            model_type="knn",
            normalized=normalized,
        )

    # Load a pickle containing a trained model and test it on the input data file
    if args.load_pickled_model:
        run_best_model(
            hive_dfs,
            top_columns_to_use,
            grid_search_output_columns,
            normalized=normalized,
            load_pickle=args.load_pickled_model,
        )

    # This is a file containing all trend columns so they don't have to
    # be generated on the fly each time for the features selection operations.
    # Generating them takes about 40 minutes since it's currently not optimized.
    trend_col_file = "trend_columns_added.csv"

    # Perform Recursive Feature Elimination to select which features
    # should be used and which ones eliminated.
    if args.recursive_feature_elimination:
        print("Performing Recursive Feature Elimination")
        # The start time of the RFE process
        start_time_raw = datetime.now()
        start_time = start_time_raw.strftime("%Y-%m-%d_%H:%M:%S")
        print(f"First Start Time: {start_time}")

        if not os.path.exists(trend_col_file):
            # This is a very slow operation
            # Generate trend columns for the columns it makes sense to have trend for
            desired_trend_cols = list(hive_dfs[0].columns)
            desired_trend_cols.remove("Month")
            desired_trend_cols.remove("Hour")
            desired_trend_cols.remove("TotalMotion")
            trend_column_names_dict = add_trend_columns(hive_dfs, desired_trend_cols)
            # Save off the file so it doesn't have to be generated later
            hive_dfs[0].to_csv(trend_col_file)
        else:
            # Load the file with all the trend columns added
            print("Loading a file with the trend columns pre-calculated")
            hive_dfs[0] = pd.read_csv(
                trend_col_file,
                header=0,
                index_col=["Time"],
                parse_dates=True,
                infer_datetime_format=True,
            )

        perform_feature_selection(
            hive_dfs,
            args.recursive_feature_elimination,
            args.input_file,
            grid_search_output_columns,
            method="rfe",
            time_groupings=[0.25],
            normalized=normalized,
        )

        end_time_raw = datetime.now()
        end_time = end_time_raw.strftime("%Y-%m-%d_%H:%M:%S")
        print(f"Last End Time: {end_time}")
        print(f"Overarching Total Time: {end_time_raw - start_time_raw}")

    # Perform Sequential Feature Selection to select which features
    # should be used and which ones eliminated.
    if args.sequential_feature_selection:
        print("Performing Sequential Feature Selection")
        # The start time of the SFS process
        start_time_raw = datetime.now()
        start_time = start_time_raw.strftime("%Y-%m-%d_%H:%M:%S")
        print(f"First Start Time: {start_time}")

        if not os.path.exists(trend_col_file):
            # This is a very slow operation
            # Generate trend columns for the columns it makes sense to have trend for
            desired_trend_cols = list(hive_dfs[0].columns)
            desired_trend_cols.remove("Month")
            desired_trend_cols.remove("Hour")
            desired_trend_cols.remove("TotalMotion")
            trend_column_names_dict = add_trend_columns(hive_dfs, desired_trend_cols)
            # Save off the file so it doesn't have to be generated later
            hive_dfs[0].to_csv(trend_col_file)
        else:
            # Load the file with all the trend columns added
            print("Loading a file with the trend columns pre-calculated")
            hive_dfs[0] = pd.read_csv(
                trend_col_file,
                header=0,
                index_col=["Time"],
                parse_dates=True,
                infer_datetime_format=True,
            )

        perform_feature_selection(
            hive_dfs,
            args.sequential_feature_selection,
            args.input_file,
            grid_search_output_columns,
            method="sfs",
            time_groupings=[0.25],
            normalized=normalized,
        )

        end_time_raw = datetime.now()
        end_time = end_time_raw.strftime("%Y-%m-%d_%H:%M:%S")
        print(f"Last End Time: {end_time}")
        print(f"Overarching Total Time: {end_time_raw - start_time_raw}")


if __name__ == "__main__":
    main()
