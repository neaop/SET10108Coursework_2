import os


# Add the number cores to start of csv file.
def prepend_cores(file_name, core_num):
    # Read in entire file.
    with open(file_name, 'r') as original:
        data = original.read()
    # Rewrite file with number of cores.
    with open(file_name, 'w') as modified:
        modified.write("Cores," + str(core_num) + '\n' + data)


# Default to 4 cores.
cores = 4
# For every file in directory.
for csv_file_name in os.listdir(os.getcwd()):
    # Skip if not a csv.
    if not csv_file_name.endswith('.csv'):
        continue
    # Split name on dashes.
    name = csv_file_name.split("_")
    # If a MPI result.
    if name[0] == "parallelMPI":
        # Get number of hosts.
        mpi_parameters = name[1].split('H')
        hosts = mpi_parameters[0]
        # Multiply hosts by 4 for cores.
        cores = int(hosts) * 4
    # If OMP + MPI result.
    elif name[0] == "parallelOMPI":
        # Get number of hosts.
        mpi_parameters = name[2].split('H')
        hosts = mpi_parameters[0]
        # Multiply hosts by 4 for cores.
        cores = int(hosts) * 4
    # Rewrite files.
    prepend_cores(csv_file_name, cores)
