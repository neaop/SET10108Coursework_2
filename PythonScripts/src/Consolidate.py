import os
import csv


consolidated_data = []

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

    row = name
    row.append(str(mean))
    row.append(str(cores))

    consolidated_data.append(row)
    csvFile.close()

with open("consolidated_data.csv", "w") as csv_file:
    writer = csv.writer(csv_file, delimiter=',', lineterminator='\n')
    for line in consolidated_data:
        writer.writerow(line)
csv_file.close()
