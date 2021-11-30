# A helper script to perform all of the file combination commands.
#
# This assumes the following file structure:
#
# temp_data_files/
# ├── combined
# │   ├── station1
# │   └── station2
# ├── dpiv
# │   ├── R_4_10_2020.csv
# │   ├── R_4_11_2020.csv
# │   ├── R_4_14_2020.csv
# │   ├── R_4_5_2020.csv
# │   ├── R_4_7_2020.csv
# │   └── R_4_8_2020.csv
# ├── usu_solar
# │   └── 05-16-2020--to--10-27-2020-usu-solar.csv
# └── weather
#     ├── station1_2020-05-16-to-2020-10-25_preprocessed.csv
#     └── station2_2020-05-19-to-2020-10-26_preprocessed.csv
#
# The Python virtual environment required by the scripts must be enabled,
# and Python3 must be installed.
#
# The file structure when complete will be as follows:
#
# temp_data_files/
# ├── combined
# │   ├── station1
# │   │   ├── R_4_10_s1_2020.csv
# │   │   ├── R_4_10_s1_2020_no_usu.csv
# │   │   ├── R_4_11_s1_2020.csv
# │   │   ├── R_4_11_s1_2020_no_usu.csv
# │   │   ├── R_4_14_s1_2020.csv
# │   │   ├── R_4_14_s1_2020_no_usu.csv
# │   │   ├── R_4_5_s1_2020.csv
# │   │   ├── R_4_5_s1_2020_no_usu.csv
# │   │   ├── R_4_7_s1_2020.csv
# │   │   ├── R_4_7_s1_2020_no_usu.csv
# │   │   ├── R_4_8_s1_2020.csv
# │   │   └── R_4_8_s1_2020_no_usu.csv
# │   └── station2
# │       ├── R_4_10_s2_2020.csv
# │       ├── R_4_10_s2_2020_no_usu.csv
# │       ├── R_4_11_s2_2020.csv
# │       ├── R_4_11_s2_2020_no_usu.csv
# │       ├── R_4_14_s2_2020.csv
# │       ├── R_4_14_s2_2020_no_usu.csv
# │       ├── R_4_5_s2_2020.csv
# │       ├── R_4_5_s2_2020_no_usu.csv
# │       ├── R_4_7_s2_2020.csv
# │       ├── R_4_7_s2_2020_no_usu.csv
# │       ├── R_4_8_s2_2020.csv
# │       └── R_4_8_s2_2020_no_usu.csv
# ├── dpiv
# │   ├── R_4_10_2020.csv
# │   ├── R_4_11_2020.csv
# │   ├── R_4_14_2020.csv
# │   ├── R_4_5_2020.csv
# │   ├── R_4_7_2020.csv
# │   └── R_4_8_2020.csv
# ├── usu_solar
# │   └── 05-16-2020--to--10-27-2020-usu-solar.csv
# └── weather
#     ├── station1_2020-05-16-to-2020-10-25_preprocessed.csv
#     └── station2_2020-05-19-to-2020-10-26_preprocessed.csv
#
# The final desired products are the files in station1 and station2, without
# the '_no_usu' part in the name.

set -eux

## Combine the data files for the Station 1 data

# R_4_5
python3 combine_data.py -we temp_data_files/weather/station1_2020-05-16-to-2020-10-25_preprocessed.csv -c temp_data_files/dpiv/R_4_5_2020.csv -o temp_data_files/combined/station1/R_4_5_s1_2020_no_usu.csv

python3 add_usu_solar_radiation.py -d temp_data_files/combined/station1/R_4_5_s1_2020_no_usu.csv -s temp_data_files/usu_solar/05-16-2020--to--10-27-2020-usu-solar.csv -o temp_data_files/combined/station1/R_4_5_s1_2020.csv

# R_4_7
python3 combine_data.py -we temp_data_files/weather/station1_2020-05-16-to-2020-10-25_preprocessed.csv -c temp_data_files/dpiv/R_4_7_2020.csv -o temp_data_files/combined/station1/R_4_7_s1_2020_no_usu.csv

python3 add_usu_solar_radiation.py -d temp_data_files/combined/station1/R_4_7_s1_2020_no_usu.csv -s temp_data_files/usu_solar/05-16-2020--to--10-27-2020-usu-solar.csv -o temp_data_files/combined/station1/R_4_7_s1_2020.csv

# R_4_8
python3 combine_data.py -we temp_data_files/weather/station1_2020-05-16-to-2020-10-25_preprocessed.csv -c temp_data_files/dpiv/R_4_8_2020.csv -o temp_data_files/combined/station1/R_4_8_s1_2020_no_usu.csv

python3 add_usu_solar_radiation.py -d temp_data_files/combined/station1/R_4_8_s1_2020_no_usu.csv -s temp_data_files/usu_solar/05-16-2020--to--10-27-2020-usu-solar.csv -o temp_data_files/combined/station1/R_4_8_s1_2020.csv

# R_4_10
python3 combine_data.py -we temp_data_files/weather/station1_2020-05-16-to-2020-10-25_preprocessed.csv -c temp_data_files/dpiv/R_4_10_2020.csv -o temp_data_files/combined/station1/R_4_10_s1_2020_no_usu.csv

python3 add_usu_solar_radiation.py -d temp_data_files/combined/station1/R_4_10_s1_2020_no_usu.csv -s temp_data_files/usu_solar/05-16-2020--to--10-27-2020-usu-solar.csv -o temp_data_files/combined/station1/R_4_10_s1_2020.csv

# R_4_11
python3 combine_data.py -we temp_data_files/weather/station1_2020-05-16-to-2020-10-25_preprocessed.csv -c temp_data_files/dpiv/R_4_11_2020.csv -o temp_data_files/combined/station1/R_4_11_s1_2020_no_usu.csv

python3 add_usu_solar_radiation.py -d temp_data_files/combined/station1/R_4_11_s1_2020_no_usu.csv -s temp_data_files/usu_solar/05-16-2020--to--10-27-2020-usu-solar.csv -o temp_data_files/combined/station1/R_4_11_s1_2020.csv

# R_4_14
python3 combine_data.py -we temp_data_files/weather/station1_2020-05-16-to-2020-10-25_preprocessed.csv -c temp_data_files/dpiv/R_4_14_2020.csv -o temp_data_files/combined/station1/R_4_14_s1_2020_no_usu.csv

python3 add_usu_solar_radiation.py -d temp_data_files/combined/station1/R_4_14_s1_2020_no_usu.csv -s temp_data_files/usu_solar/05-16-2020--to--10-27-2020-usu-solar.csv -o temp_data_files/combined/station1/R_4_14_s1_2020.csv


## Combine the data files for the Station 2 data

# R_4_5
python3 combine_data.py -we temp_data_files/weather/station2_2020-05-19-to-2020-10-26_preprocessed.csv -c temp_data_files/dpiv/R_4_5_2020.csv -o temp_data_files/combined/station2/R_4_5_s2_2020_no_usu.csv

python3 add_usu_solar_radiation.py -d temp_data_files/combined/station2/R_4_5_s2_2020_no_usu.csv -s temp_data_files/usu_solar/05-16-2020--to--10-27-2020-usu-solar.csv -o temp_data_files/combined/station2/R_4_5_s2_2020.csv

# R_4_7
python3 combine_data.py -we temp_data_files/weather/station2_2020-05-19-to-2020-10-26_preprocessed.csv -c temp_data_files/dpiv/R_4_7_2020.csv -o temp_data_files/combined/station2/R_4_7_s2_2020_no_usu.csv

python3 add_usu_solar_radiation.py -d temp_data_files/combined/station2/R_4_7_s2_2020_no_usu.csv -s temp_data_files/usu_solar/05-16-2020--to--10-27-2020-usu-solar.csv -o temp_data_files/combined/station2/R_4_7_s2_2020.csv

# R_4_8
python3 combine_data.py -we temp_data_files/weather/station2_2020-05-19-to-2020-10-26_preprocessed.csv -c temp_data_files/dpiv/R_4_8_2020.csv -o temp_data_files/combined/station2/R_4_8_s2_2020_no_usu.csv

python3 add_usu_solar_radiation.py -d temp_data_files/combined/station2/R_4_8_s2_2020_no_usu.csv -s temp_data_files/usu_solar/05-16-2020--to--10-27-2020-usu-solar.csv -o temp_data_files/combined/station2/R_4_8_s2_2020.csv

# R_4_10
python3 combine_data.py -we temp_data_files/weather/station2_2020-05-19-to-2020-10-26_preprocessed.csv -c temp_data_files/dpiv/R_4_10_2020.csv -o temp_data_files/combined/station2/R_4_10_s2_2020_no_usu.csv

python3 add_usu_solar_radiation.py -d temp_data_files/combined/station2/R_4_10_s2_2020_no_usu.csv -s temp_data_files/usu_solar/05-16-2020--to--10-27-2020-usu-solar.csv -o temp_data_files/combined/station2/R_4_10_s2_2020.csv

# R_4_11
python3 combine_data.py -we temp_data_files/weather/station2_2020-05-19-to-2020-10-26_preprocessed.csv -c temp_data_files/dpiv/R_4_11_2020.csv -o temp_data_files/combined/station2/R_4_11_s2_2020_no_usu.csv

python3 add_usu_solar_radiation.py -d temp_data_files/combined/station2/R_4_11_s2_2020_no_usu.csv -s temp_data_files/usu_solar/05-16-2020--to--10-27-2020-usu-solar.csv -o temp_data_files/combined/station2/R_4_11_s2_2020.csv

# R_4_14
python3 combine_data.py -we temp_data_files/weather/station2_2020-05-19-to-2020-10-26_preprocessed.csv -c temp_data_files/dpiv/R_4_14_2020.csv -o temp_data_files/combined/station2/R_4_14_s2_2020_no_usu.csv

python3 add_usu_solar_radiation.py -d temp_data_files/combined/station2/R_4_14_s2_2020_no_usu.csv -s temp_data_files/usu_solar/05-16-2020--to--10-27-2020-usu-solar.csv -o temp_data_files/combined/station2/R_4_14_s2_2020.csv

