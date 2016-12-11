import os
import csv
import re

file_name = ""
consolidated_data = []
csv_headers = []

for csv_file_name in os.listdir(os.getcwd()):
    # Skip non csv files.
    if not csv_file_name.endswith('.csv'):
        continue

    # Open csv file
    csv_file = open(csv_file_name)
    reader = csv.reader(csv_file)
    csv_data = []
    # Read number of cores from first line.
    cores = int(csv_file.readline().split(',')[1])
    # Read open csv.
    for row in reader:
        csv_data.append(int(row[1]))

    # If contains more than 100 samples (mpi).
    while len(csv_data) > 100:
        # Remove samples.
        csv_data.pop(0)

    # Calculate mean.
    mean = (sum(csv_data) / len(csv_data))
    # Split file name on underscore
    name = csv_file_name.split('_')
    # Remove identifier
    name.pop()
    # Set the file name to the method type.
    file_name = name[0]

    if name[0] == "sequential":
        csv_headers = ["Method", "Samples", "Average Time", "Cores"]
        # Remove letters from value.
        name[1] = (re.sub(r'[A-Z]', '', name[1]))

    elif name[0] == "parallelOMP":
        csv_headers = ["Method", "Configuration", "Samples", "Average Time", "Cores"]
        # Remove letters from value.
        name[2] = (re.sub(r'[A-Z]', '', name[2]))

    elif name[0] == "parallelOMPI":
        csv_headers = ["Method", "Configuration", "Hosts", "Nodes", "Samples", "Average Time", "Cores"]
        # Separate hosts from nodes.
        mpi = re.split(r'[\D]', name[2])
        mpi = list(filter(None, mpi))
        name[2] = mpi[0]
        name.insert(3, mpi[1])
        # Remove letters from value.
        name[4] = (re.sub(r'[A-Z]', '', name[4]))

    else:
        csv_headers = ["Method", "Hosts", "Nodes", "Samples", "Average Time", "Cores"]
        # Separate hosts from nodes.
        mpi = re.split(r'[\D]', name[1])
        mpi = list(filter(None, mpi))
        name[1] = mpi[0]
        name.insert(2, mpi[1])
        # Remove letters from value.
        name[3] = (re.sub(r'[A-Z]', '', name[3]))

    # Combine data into one list.
    row = name
    row.append(str(mean))
    row.append(str(cores))
    consolidated_data.append(row)
    # Close file.
    csv_file.close()

# Create new csv.
with open(file_name+"_consolidated_data.csv", "w") as csv_file:
    writer = csv.writer(csv_file, delimiter=',', lineterminator='\n')
    # Write file header.
    writer.writerow(csv_headers)
    # Write each line into file.
    for line in consolidated_data:
        writer.writerow(line)
# Close file.
csv_file.close()
