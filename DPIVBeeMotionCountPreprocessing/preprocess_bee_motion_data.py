# Preprocess Bee Motion Data
#
# Consolidates the bee counts for each BeePi video into a sum of the bee
# motions for each direction as well as a total of all bee motions for
# during the video.
#
# Written by Daniel Hornberger

import argparse
import csv
import os
import fileinput
import datetime
import operator


def preprocess_bee_motion_counts(root_dir, output_filename):
    """
    Generate a .csv file containing a row for each video data file containing
    the bee motion counts for each direction as well as a total of all bee
    motions in each video.
    """

    print(f"Creating {output_filename} to hold the data...")
    temp_output_filename = ".unsorted_directional_bee_motion_counts.csv"
    with open(temp_output_filename, "w") as temp_output_file:
        # Write the labels row
        temp_output_file.write("Time,BeeMonitorID,UpwardMotion,DownwardMotion,LateralMotion,TotalMotion\n")

        # The number of rows to be discarded. This will happen usually
        # when the BeePi monitor was first started not on an even quarter hour.
        discard_count = 0

        for dir_name, subdir_list, file_list in os.walk(root_dir):
            # Sort the file list alphabetically in increasing order
            # Loop through each .csv file in the current directory
            for data_file in file_list:
                file_path = os.path.join(dir_name, data_file)
                print(f"Processing {file_path}...")

                # Parse the date from the filename
                # The resulting date needs to be in the format
                # YYYY-MM-DD HH:MM:00
                date_start = data_file.find("-") + 1
                date_end = date_start + 18  # Assuming YYYY-MM-DD_HH-MM-SS
                time = datetime.datetime.strptime(
                    data_file[date_start:date_end], "%Y-%m-%d_%H-%M-%S"
                )
                time_str = time.strftime("%Y-%m-%d %H:%M:00")
                print("\t" + time_str)

                # Discard rows with unusable times
                if time.minute % 15 != 0:
                    print(
                        "Minute value unevenly divisible by 15 detected. Discarding. "
                        "This is likely the first file of the record."
                    )
                    print(str(time))
                    discard_count = discard_count + 1
                    # Don't add this file's contents to the csv file
                    continue

                # Parse the IP station identifier from the filename
                id_start = 8  # Assuming the first part of the IP address is 192_168_
                id_end = data_file.find("-")
                monitor_id = data_file[id_start:id_end]
                monitor_id = monitor_id.replace("_", ".")
                print("\t" + monitor_id)

                # Remove the quotes around each value in the file
                print("\tRemoving the quotes around each value")
                text_to_search = '"'
                replacement_text = ""
                with fileinput.FileInput(file_path, inplace=True) as file:
                    for line in file:
                        print(line.replace(text_to_search, replacement_text), end="")

                # Calculate the sums of the 'UPWARD', 'DOWNWARD', 'LATERAL', and
                # 'TOTAL' columns
                upward_sum = 0
                downward_sum = 0
                lateral_sum = 0
                total_sum = 0
                with open(file_path, newline="") as reader_file:
                    reader = csv.DictReader(reader_file, delimiter=",")
                    for row in reader:
                        upward_sum += int(row["UPWARD"])
                        downward_sum += int(row["DOWNWARD"])
                        lateral_sum += int(row["LATERAL"])
                        total_sum += int(row["TOTAL"])

                print(f"\tUpward Sum:   {upward_sum}")
                print(f"\tDownward Sum: {downward_sum}")
                print(f"\tLateral Sum:  {lateral_sum}")
                print(f"\tTotal Sum:    {total_sum}")

                # Write the sum values, separated by commas, to the .csv file
                temp_output_file.write(
                    f"{time_str},{monitor_id},{upward_sum},{downward_sum},{lateral_sum},{total_sum}\n"
                )

    # Sort the resulting csv file
    print("Sorting the output .csv file in ascending order by date...")
    header_row = []
    sorted_csv_rows = []
    with open(temp_output_filename, newline="") as temp_output_file:
        reader = csv.reader(temp_output_file, delimiter=",")
        header_row = next(reader)

        # Sort the csv file by the 'Time' column in ascending order
        sorted_csv_rows = sorted(reader, key=operator.itemgetter(0), reverse=False)

    os.remove(temp_output_filename)

    print("Writing the final output file...")
    with open(output_filename, "w", newline="") as output_file:
        writer = csv.writer(output_file, delimiter=",")
        writer.writerow(header_row)
        for row in sorted_csv_rows:
            writer.writerow(row)

    print(f"Discarded rows with unusable timestamps: {discard_count}")
    print("Done.")


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-r",
        "--root-dir",
        action="store",
        required=True,
        dest="root_dir",
        help="The directory containing any DPIVBeeCount subdirectories or files to be processed",
    )
    parser.add_argument(
        "-o",
        "--output-file",
        action="store",
        required=True,
        dest="output_file",
        help="The name of the newly created processed file .csv file",
    )

    args = parser.parse_args()
    preprocess_bee_motion_counts(args.root_dir, args.output_file)


if __name__ == "__main__":
    main()
