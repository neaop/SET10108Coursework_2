import os
import csv
import re

filename = ""
consolidated_data = []
csv_headers = []

for csvFileName in os.listdir(os.getcwd()):
    if not csvFileName.endswith('.csv'):
        continue
    print('Accessing: ' + csvFileName)

    csvFile = open(csvFileName)
    reader = csv.reader(csvFile)
    csv_data = []
    cores = int(csvFile.readline().split(',')[1])

    for row in reader:
        csv_data.append(int(row[1]))

    while len(csv_data) > 100:
        csv_data.pop(0)

    mean = (sum(csv_data) / len(csv_data))
    name = csvFileName.split('_')
    name.pop()
    filename = name[0]

    if name[0] == "sequential":
        csv_headers = ["Method", "Samples", "Average Time", "Cores"]
        name[1] = (re.sub(r'[a-zA-Z]', '', name[1]))

    elif name[0] == "parallelOMP":
        csv_headers = ["Method", "Configuration", "Samples", "Average Time", "Cores"]
        name[2] = (re.sub(r'[a-zA-Z]', '', name[2]))

    elif name[0] == "parallelOMPI":
        csv_headers = ["Method", "Configuration", "Hosts", "Nodes", "Samples", "Average Time", "Cores"]

        mpi = re.split(r'[\D]', name[2])
        mpi = list(filter(None, mpi))
        name[2] = mpi[0]
        name.insert(3, mpi[1])

        name[4] = (re.sub(r'[a-zA-Z]', '', name[4]))

    else:
        csv_headers = ["Method", "Hosts", "Nodes", "Samples", "Average Time", "Cores"]

        mpi = re.split(r'[\D]', name[1])
        mpi = list(filter(None, mpi))
        name[1] = mpi[0]
        name.insert(2, mpi[1])

        name[3] = (re.sub(r'[a-zA-Z]', '', name[3]))

    print(csv_headers)
    print(name)
    row = name
    row.append(str(mean))
    row.append(str(cores))
    consolidated_data.append(row)
    csvFile.close()

with open(filename+"_consolidated_data.csv", "w") as csv_file:
    writer = csv.writer(csv_file, delimiter=',', lineterminator='\n')
    for line in consolidated_data:
        writer.writerow(line)
csv_file.close()
