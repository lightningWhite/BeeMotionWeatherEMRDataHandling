# The purpose of this script is to fix timestamps that were behind the actual
# time after a power outage.

import csv
from datetime import datetime
from datetime import timedelta
import fileinput
import os
import shutil


def create_preprocessed_file(filename, preprocessed_filename):
    """
    Make a copy of the file and append a .bak to it's name.
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


def adjust_timestamps(filename, minute_adjustment):
    """
    Round the timestamps in a specified csv file to the nearest quarter hour
    interval starting from the hour (00, 15, 30, 45 minute endings).
    """
    print(f"Adding {minute_adjustment} minutes to the timestamps...")

    # Make a temporary copy of the data file to read
    temp_filename = ".temp_reader_file"
    shutil.copyfile(filename, temp_filename)
    with open(temp_filename, newline="") as temp_reader_file:
        reader = csv.DictReader(temp_reader_file, delimiter=",")

        with open(filename, "w", newline="") as csvfile_preprocessed:
            writer = csv.DictWriter(csvfile_preprocessed, fieldnames=reader.fieldnames)
            writer.writeheader()

            for row in reader:
                print(f"Printing the row: {row}")
                timestamp = datetime.strptime(row["Time"], "%Y-%m-%d %H:%M:%S.%f")

                # Adjust the time by the minute_adjustment
                adjusted_time = timestamp + timedelta(
                    minutes=minute_adjustment, seconds=00, microseconds=000000
                )

                print(
                    f"Adjusting {timestamp.strftime('%Y-%m-%d %H:%M:%S.%f')} "
                    "to {adjusted_time.strftime('%Y-%m-%d %H:%M:%S.%f')}"
                )

                # Write the row with the rounded time to the new csv file
                row["Time"] = adjusted_time
                writer.writerow(row)

    os.remove(temp_filename)

    print("Done adjusting the timestamps.")


filename = "./07-31-2020--10-17-08.csv"
# The number of minutes to add or subtract from the timestamps
preprocessed_filename = "07-31-2020--10-17-08--adjusted.csv"

minute_adjustment = 54

create_preprocessed_file(filename, preprocessed_filename)
remove_spaces_after_commas(preprocessed_filename)
adjust_timestamps(preprocessed_filename, minute_adjustment)
print("Done.")
