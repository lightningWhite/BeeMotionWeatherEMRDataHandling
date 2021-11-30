# Bee Motion, Weather, and EMF Data Handling

This repository contains python scripts used for preprocessing the .csv files generated by the WeatherAndEMFSensingStation as well as the bee counts obtained from the BeePi monitors using the DPIV Bee Motion Counting algorithm.
It also contains scripts for combining the preprocessed DPIV Bee Motion count files with the WeatherEMF files. Additionally, scripts are provided for exploring, analyzing, and performing other operations on the combined Weather, EMF, and Bee Motion Count data.

The general process for processing the data and preparing it for analysis is as follows with commands in their respective directories as examples:

### 1. Preprocess the WeatherAndEMF Data

Here is an example usage of the preprocess_weather_data.py script:

```
python3 preprocess_weather_data.py -i /path/to/WeatherAndEMFSensingData/05-16-2020--16-23-44-to-10-26--04-52-11.csv -o /path/to/ProcessedWeatherAndEMFData/05-16-2020--16-23-44-to-10-26--04-52-11--preprocessed.csv
```

### 2. Preprocess the Bee Motion Data

Here is an example usage of the preprocess_bee_motion_data.py script operating on one of the BeePi monitor directories (R_4_5):

```
python3 preprocess_bee_motion_data.py -r /path/to/DPIV_Bee_Motion_Counts_Season2020/R_4_5 -o /path/to/ProcessedDPIVBeeCountData/R_4_5_2020.csv
```

### 3. Combine the WeatherAndEMF Data and the Bee Motion Data

Here is an example usage of the combine_data.py script operating on one of the preprocessed BeePi monitor files and the preprocessed WeatherEMF data files:

```
python3 combine_data.py -we /path/to/ProcessedWeatherAndEMFData/05-16-2020--16-23-44-to-10-26--04-52-11--preprocessed.csv -c /path/to/ProcessedDPIVBeeCountData/R_4_5_2020.csv -o /path/to/CombinedProcessedData/R_4_5_2020_combined.csv
```

### 4. Add the Solar Radiation USU (MJ/m^2) Column to the Combined Data

Since the Solar Radiation data collected by the weather stations deployed at the hive site isn't ideal due to the shadows from the trees above, supplemental Solar Radiation data was obtained from one of USU Climate Center's weather
stations.
This data can be added as an additional column to the Combined Preprocessed Data with the add_usu_solar_radiation.py script with a command such as the following:

```
python3 add_usu_solar_radiation.py -d /path/to/CombinedProcessedData/R_4_5_2020_combined.csv -s /path/to/05-16-2020--to--10-27-2020-usu-solar.csv -o /path/to/output_file.csv
```

### 5. Calibrate the Data

Note: This was not used, but is left here in case it becomes desired.
Data calibration became much more involved due to the EMF-390 sensors.

If the continuous data file contains data from two different EMF-390 sensors, it's likely that the data needs to be calibrated between them.
This is due to the readings from one sensor being quite a bit higher or lower than the other.
The `calibrate_combined_data.py` script will use the second Station ID to calculate a line of best fit.
Then it will use the slope and offset to correct the values from the first station.

```
python3 calibrate_combined_data.py -i R_4_5_2020_combined_with_usu_solar_uncalibrated.csv -o R_4_5_2020_calibrated.csv
```

### 6. Analyze the Data

Here is _one_ example usage of the data_exploration.py script operating on the combined preprocessed WeatherEMF and Bee Motion data after creating the 'output' directory:

```
python3 data_exploration.py -i R_4_5_2020_combined.csv -o output -a
```

### 7. Random Forest Grid Search the Original Columns

This will perform a grid search of training and testing random forest models using different combinations of the base columns.
Various statistics will be calculated for each model, including the R^2 score, AICc score, feature importances, etc.
This test may take several hours to complete.

Example usage:

```
python3 main_experiments.py -i R_4_5_2020_calibrated.csv --orig-col-grid-search output/directory/path
```

### 8. Select the Best Model Obtained from the Original Column Grid Search

This will place the best performing model as the first row in the output file to know what combination of columns should be used,
how it performs, how likely it is to be the best model than the others, the feature importances, etc.
The resulting model from this selection will be used as the base model when adding additional columns.

Example usage:

```
python3 model_selection.py -i original_col_grid_search_results.csv -o model_selection_results.csv
```

Update the `main_experiments.py` script with the top selected columns.

### 9. Random Forest Grid Search the Original Columns with Nights Removed

This was not used, but is left here for reference if desired.

This will do the same as the previous grid search, but the nights will be removed.
This can be used to determine whether the RF columns have greater feature importances with nights included than when they aren't, since RF travels better at night.

Example usage:

```
python3 main_experiments.py -i R_4_5_2020_calibrated.csv --orig-col-grid-search-nights-excluded output/directory/path
```

### 10. Select the Best Model Obtained from the Original Column Grid Search with Nights Removed

This was not used, but is left here for reference if desired.

This will place the best performing model as the first row in the output file to know what combination of columns should be used,
how it performs, how likely it is to be the best model than the others, the feature importances, etc.
The resulting model from this selection will be compared against the resulting model from the grid search with nights included.
This allows us to determine whether the nights are causing the RF columns to rise in feature importance.

Example usage:

```
python3 model_selection.py -i original_col_grid_search_nights_removed_results.csv -o model_selection_nights_removed_results.csv
```

Update the `main_experiments.py` script with the top selected columns.

### 11. Compare the Importance of RF With and Without Nights

Compare the top 50 models of each of the model selection results files of when nights were included and when they were excluded.
Count how many included any RF or other EMR values from each, and see if there are fewer models using RF when nights are removed than when they are included.
If there are fewer, then it's likely that the inclusion of RF is due to the diurnal cycles of RF since RF travels better at night than during the day - which creates a negative correlation between it and bee motion.
If RF is still an important feature, it could possibly still be due to the diurnal cycles, but it can't be ruled out as a variable that can be used in predicting bee total motion.

### 12. Trend Column Additive Analysis

This will use the best model previously obtained by the grid search on the base columns with nights included.
This creates 24 new columns for each of the selected top performing columns.
Each new column contains the trend of the values spaning 1 to 24 hours.
A model is trained for each trend column for each variable, and a separate file is produced containing the results for each base column and its associated trend columns.
Later, the best trend column will be selected.

```
python3 main_experiments.py -i R_4_5_2020_calibrated.csv --trend-column-additive-analysis output/directory/path
```

### 13. Select the Best Trend Interval for Each Column

This will place the best performing model as the first row in the output file to know what trend interval is optimal for a column.
Run this on each column's additive analysis file produced from the previous step.

Example usage:

```
python3 model_selection.py -d directory/to/additive/analysis/files
```

Update the `main_experiments.py` script with the top selected columns.

### 15. Grid Search the Combinations Top Trend Columns While Using the Previously Selected Base Columns

This will allow the selection of which trend columns to use, appended to the base columns.


Example usage:

```
python3 main_experiments.py -i R_4_5_2020_calibrated.csv --trend-column-grid-search output/directory/path
```

### 16. Select the Best Model Obtained from the Original Column Grid Search and the Trend Column Grid Search

This will place the best performing model as the first row in the output file to know the final combination of columns that should be used.
Before performing this, copy the top model selected from the base column grid search into the resulting file from the top trend column grid search.
This will prove that adding the trend columns is better than the models without them (if there is a model that performs better with the trend columns).

Example usage:

```
python3 model_selection.py -i trend_column_grid_search_output.csv -o trend_column_grid_search_output_model_selection.csv
```

Update the `main_experiments.py` script with the top selected columns added to the selected base columns.

## 17. Run the Random Forest Hyperparameter Grid Search

This will select the best hyperparameters of the Random Forest to use while working on the top column combinations.

Example usage:

```
python3 main_experiments.py -i R_4_5_2020_calibrated.csv --hyperparameter-grid-search output/directory/path
```

Update the `main_experiments.py` script with the top selected hyperparameters.

## 18. Run the Time Grouping Tests

This will train models on the data when it is grouped from 15 minute intervals to 24 hour intervals to show the time grouping interval with the highest performing accuracy.

Example usage:

```
python3 main_experiments.py -i R_4_5_2020_calibrated.csv --time-grouping-tests output/directory/path
```

## 19. Run the Best Model

After the previous steps have been run, the best model can be used by running the following:

Example usage:

```
python3 main_experiments.py -i R_4_5_2020_calibrated.csv --best-model
```

## Script Documentation

### WeatherDataPreprocessing/

This directory contains scripts for preprocessing the data obtained from the Weather and EMF sensing station.

Here is an example of the raw Weather and EMF data the script can operate on:

```
Record Number, Time, Temperature (F), Pressure (mbar), Humidity (%), Wind Direction (Degrees), Wind Direction (String), Wind Speed (MPH), Wind Gust (MPH), Precipitation (Inches), Shortwave Radiation (W m^(-2)), Avg. RF Watts (W), Avg. RF Watts Frequency (MHz), Peak RF Watts (W), Frequency of RF Watts Peak (MHz), Peak RF Watts Frequency (MHz), Watts of RF Watts Frequency Peak (W), Avg. RF Density (W m^(-2)), Avg. RF Density Frequency (MHz), Peak RF Density (W m^(-2)), Frequency of RF Density Peak (MHz), Peak RF Density Frequency (MHz), Density of RF Density Frequency Peak (W m^(-2)), Avg. Total Density (W m^(-2)), Max Total Density (W m^(-2)), Avg. EF (V/m), Max EF (V/m), Avg. EMF (mG), Max EMF (mG),Station ID
1, 2020-05-16 16:38:54.100572,67.8,1014.3,23.8,256.5, WSW,0.7,2.5,0,130,3.326912E-10,694,5E-09,1866,1866,5E-09,3.38235294118E-05,732.6,0.0005,1866,1881,0.0003,0.000126470588235,0.001,693.7,886,1.2,1.4,1
2, 2020-05-16 16:53:55.786168,66.2,1014.3,24,252, WSW,0.5,2.1,0,140,8.500571E-10,759.1,2.5E-08,1880,1881,4E-09,5.42857142857E-05,776.4,0.0013,1880,1881,0.0004,0.000301428571429,0.0011,676.6,844,1.3,1.5,1
3, 2020-05-16 17:08:56.629817,65.8,1014.1,23.3,255.5, WSW,0.4,3.1,0,130,4.531143E-10,741,5E-09,1861,1878,3E-09,2.85714285714E-05,758.6,0.0005,1861,1878,0.0003,0.00026,0.0011,690.1,889.6,1.2,1.4,1
4, 2020-05-16 17:23:58.887251,65.1,1014.2,24.5,248.4, WSW,0.3,1.5,0,140,5.634E-10,775.9,1.6E-08,1884,1884,1.6E-08,2.8571428571E-06,671.5,0.0001,1880,1880,0.0001,0.000472857142857,0.0028,664.1,880,1.2,1.5,1
5, 2020-05-16 17:38:59.409071,65.1,1014.2,23.7,295.7, WNW,0.1,0.9,0,145,2.303E-10,670.8,3E-09,1872,1872,3E-09,2.28571428571E-05,776,0.0003,1872,1879,0.0003,0.001105714285714,0.0029,675.8,880,1.2,1.4,1
```

And here is an example of the preprocessed data after the script has operated on it:

```
Record Number,Time,Temperature (F),Pressure (mbar),Humidity (%),Wind Direction (Degrees),Wind Direction (String),Wind Speed (MPH),Wind Gust (MPH),Precipitation (Inches),Shortwave Radiation (W m^(-2)),Avg. RF Watts (W),Avg. RF Watts Frequency (MHz),Peak RF Watts (W),Frequency of RF Watts Peak (MHz),Peak RF Watts Frequency (MHz),Watts of RF Watts Frequency Peak (W),Avg. RF Density (W m^(-2)),Avg. RF Density Frequency (MHz),Peak RF Density (W m^(-2)),Frequency of RF Density Peak (MHz),Peak RF Density Frequency (MHz),Density of RF Density Frequency Peak (W m^(-2)),Avg. Total Density (W m^(-2)),Max Total Density (W m^(-2)),Avg. EF (V/m),Max EF (V/m),Avg. EMF (mG),Max EMF (mG),Station ID
1,2020-05-16 16:45:00,67.8,1014.3,23.8,256.5,WSW,0.7,2.5,0,130,3.326912E-10,694,5E-09,1866,1866,5E-09,3.38235294118E-05,732.6,0.0005,1866,1881,0.0003,0.000126470588235,0.001,693.7,886,1.2,1.4,1
2,2020-05-16 17:00:00,66.2,1014.3,24,252,WSW,0.5,2.1,0,140,8.500571E-10,759.1,2.5E-08,1880,1881,4E-09,5.42857142857E-05,776.4,0.0013,1880,1881,0.0004,0.000301428571429,0.0011,676.6,844,1.3,1.5,1
3,2020-05-16 17:15:00,65.8,1014.1,23.3,255.5,WSW,0.4,3.1,0,130,4.531143E-10,741,5E-09,1861,1878,3E-09,2.85714285714E-05,758.6,0.0005,1861,1878,0.0003,0.00026,0.0011,690.1,889.6,1.2,1.4,1
4,2020-05-16 17:30:00,65.1,1014.2,24.5,248.4,WSW,0.3,1.5,0,140,5.634E-10,775.9,1.6E-08,1884,1884,1.6E-08,2.8571428571E-06,671.5,0.0001,1880,1880,0.0001,0.000472857142857,0.0028,664.1,880,1.2,1.5,1
5,2020-05-16 17:45:00,65.1,1014.2,23.7,295.7,WNW,0.1,0.9,0,145,2.303E-10,670.8,3E-09,1872,1872,3E-09,2.28571428571E-05,776,0.0003,1872,1879,0.0003,0.001105714285714,0.0029,675.8,880,1.2,1.4,1
```

### preprocess_weather_data.py

In this directory, there is a python3 script named `preprocess_weather_data.py`.
This script is intended to clean up and prepare the weather and EMF sensing data for pairing with the bee motion count data. It performs the following operations on a file generated by the Weather and EMF sensing station:

* Removes the random spaces after commas in the file
* Rounds the timestamp of each record to the nearest quarter hour
* Creates a new file with the modified data

The file on which the script will operate as well as the processed output
file can be specified as command line arguments to the script.

See the help text by running `python3 preprocess_weather_data.py --help`.

### DPIVBeeMotionCountPreprocessing/

In this directory, there is a python3 script named `preprocess_bee_motion_data.py`.
This script is intented to consolidate the bee counts for each BeePi video into a sum of bee motions for each direction as well as a total of all bee motions for a video.
This is done so this bee motion count data can be paired up with the weather and EMF sensing data.
Note that data for different monitoring stations should be processed separately since they will have overlapping time stamps.

This script performs the following operations on each .csv file in a directory and its subdirectories as outputted by the DPIV bee motion counting algorithm (that are renamed as described above):

* Parses the date and time from the name of the .csv file
* Parses the last two numbers of the IP address in the file as the ID of the monitor that gathered the data
* Removes the quotes around each value in the file
* Calculates the sums of the Upward, Downward, Lateral, and Total columns
* Writes the date/time and sums as a row to a .csv file that will contain a row for each .csv file that is processed

The directory on which to operate as well as the name of the output file can be specified as command line arguments to the script.

See the help text by running `python3 preprocess_bee_motion_data.py --help` for more information.

Also note that this data contains gaps.
The BeePi monitor only operates from 7:00 AM until Midnight.
The `combine_motion_data_with_weather_and_emf_data.py`script will inflate the data for the missing hours and set the counts to zero for those rows so it pairs up better with the weather and EMF sensing data.

Here is an example of the raw DPIV Bee Motion data the script can operate on:

```
FRAME,UPWARD,DOWNWARD,LATERAL,TOTAL,TOTAL_THRESHOLD,TOTAL_ARRAY_COUNT
16,1,1,2,4,0.8,35
17,0,1,3,4,0.8,35
18,2,2,2,6,1.2000000000000002,35
19,0,0,0,0,0.0,35
20,0,2,0,2,0.4,35
```

And here is an example of the preprocessed data after the script has operated on it:

```
Time,BeeMonitorID,UpwardMotion,DownwardMotion,LateralMotion,TotalMotion
2020-04-30 10:45:00,4.5,469,492,422,1383
2020-04-30 11:00:00,4.5,772,790,977,2539
2020-04-30 11:15:00,4.5,290,303,128,721
2020-04-30 11:30:00,4.5,536,541,444,1521
2020-04-30 11:45:00,4.5,627,566,377,1570
```

#### DPIV Bee Count File Notes

The DPIV Bee Counts consists of directories representing a different data collection day from the BeePi monitors. 
Each gathering day is approximately two weeks apart. 
The directories are named as follows:

```
R_<Second-to-LastIP#>_<LastIP#>_<DayMonthYear Gathered>
```

For example:

```
R_4_5_16May2020
```

The files inside each directory are named using the following convention:

```
<IP Address>-<YYYY-MM-DD>_<DPIV Parameters>_bee_motion_counts.csv
```

#### Data Files

Each data file contains the following information in a Comma Separated Value format:

```
"FRAME","UPWARD","DOWNWARD","LATERAL","TOTAL","TOTAL_THRESHOLD","TOTAL_ARRAY_COUNT"
```

Each file contains the DPIV bee motion count data for a single video taken at the time indicated in the file name.

A description of the fields is as follows:

* Frame - The frame on which the algorithm was performed
  * Note that it takes two frames to perform the DPIV
* Upward - A bee motion that is moving away from the hive
* Downward - A bee motion that is moving toward the hive
* Lateral - A bee motion that is moving parallel to the entrance of the hive
* Total - The sum of the three different types of motions
* Total Threshold - ???
* Total Array Count - ???

Note that the frames start at 16 rather than 0 or 1.
This is because the camera produce poor imagery for the first few frames of each video.
The first few frames as well as the last few were discarded.

Some changes were recently made to the DPIV bee motion counting algorithm that made it so the values reported were the number of bee motions between two frames.
In a previous version of the algorithm, the value reported was the number of vectors in a vector field associated with a bee motion.
This vector field has been combined into a single vector.
Thus, the value reported is the number of different bees moving between the two frames.

To calculuate the number of bee motions for each individual direction, the sum of each direction column needs to be calculated individually.

### CombineData/

This directory contains a script that allows the user to specify a preprocessed weather and EMF sensing data .csv file and a preprocessed DPIV bee motion count data .csv file which get combined into a new .csv file with a name specified by the user.
The bee motion count columns are added on the right side of the weather and EMF sensing data columns.

A row is created for each timestamp in the minimum and maximum range between both files.
If there are missing rows in one of the two files, 'nan' will be used.

Here is an example of some combined preprocessed Bee Motion and Weather/EMF data:

```
Record Number,Time,Temperature (F),Pressure (mbar),Humidity (%),Wind Direction (Degrees),Wind Direction (String),Wind Speed (MPH),Wind Gust (MPH),Precipitation (Inches),Shortwave Radiation (W m^(-2)),Avg. RF Watts (W),Avg. RF Watts Frequency (MHz),Peak RF Watts (W),Frequency of RF Watts Peak (MHz),Peak RF Watts Frequency (MHz),Watts of RF Watts Frequency Peak (W),Avg. RF Density (W m^(-2)),Avg. RF Density Frequency (MHz),Peak RF Density (W m^(-2)),Frequency of RF Density Peak (MHz),Peak RF Density Frequency (MHz),Density of RF Density Frequency Peak (W m^(-2)),Avg. Total Density (W m^(-2)),Max Total Density (W m^(-2)),Avg. EF (V/m),Max EF (V/m),Avg. EMF (mG),Max EMF (mG),Station ID,BeeMonitorID,UpwardMotion,DownwardMotion,LateralMotion,TotalMotion
1,2020-05-16 16:45:00,67.8,1014.3,23.8,256.5,WSW,0.7,2.5,0,130,3.326912E-10,694,5E-09,1866,1866,5E-09,3.38235294118E-05,732.6,0.0005,1866,1881,0.0003,0.000126470588235,0.001,693.7,886,1.2,1.4,1,4.5,183,230,177,590
2,2020-05-16 17:00:00,66.2,1014.3,24,252,WSW,0.5,2.1,0,140,8.500571E-10,759.1,2.5E-08,1880,1881,4E-09,5.42857142857E-05,776.4,0.0013,1880,1881,0.0004,0.000301428571429,0.0011,676.6,844,1.3,1.5,1,4.5,106,118,168,392
3,2020-05-16 17:15:00,65.8,1014.1,23.3,255.5,WSW,0.4,3.1,0,130,4.531143E-10,741,5E-09,1861,1878,3E-09,2.85714285714E-05,758.6,0.0005,1861,1878,0.0003,0.00026,0.0011,690.1,889.6,1.2,1.4,1,4.5,179,231,16,426
4,2020-05-16 17:30:00,65.1,1014.2,24.5,248.4,WSW,0.3,1.5,0,140,5.634E-10,775.9,1.6E-08,1884,1884,1.6E-08,2.8571428571E-06,671.5,0.0001,1880,1880,0.0001,0.000472857142857,0.0028,664.1,880,1.2,1.5,1,4.5,112,161,106,379
5,2020-05-16 17:45:00,65.1,1014.2,23.7,295.7,WNW,0.1,0.9,0,145,2.303E-10,670.8,3E-09,1872,1872,3E-09,2.28571428571E-05,776,0.0003,1872,1879,0.0003,0.001105714285714,0.0029,675.8,880,1.2,1.4,1,4.5,29,43,142,214
```

### DataAnalysis/

This directory contains scripts for exploring, analyzing, and operating on the preprocessed and combined .csv files.

The dependencies must be installed before using these scripts.
To do this, perform the following steps:

Create a Python3 virtual environment (this is only needed the first time):

```
python3 -m venv env
```

Activate the virtual environment:

```
source env/bin/activate
```

Upgrade pip:

```
pip install --upgrade pip
```

Install the dependencies:

```
pip install -r requirements.txt
```

Now the following scripts can be used:

`data_exploration.py`:
- For generating various plots of or about the data

`main_experiments.py`:
- For additional data preprocessing and column addition
- For training random forest models
- For grid searching best columns to use
- For grid searching with nights removed
- For grid searching with trend columns
- For grid searching hyperparameters
- For testing different data time groupings

`model_selection.py`:
- For generating various statistics and analyses on the results of the grid searches to select the best model

`knn.py`:
- Train and test a K-Nearest-Neighbors model on the data

`ordinary_least_squares.py`:
- Train and test an Ordinary Least Squares model on the data

`combos.py`:
- A simple test script to estimate how long it will take to perform a grid search on a certain number of columns
