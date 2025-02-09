In order generate the graphs from the main body of the article (Example section) use the following command:
python main --config example1 --dest_dir DEST_DIR --n_cores N_CORES

where DEST_DIR is the destination directory to save the figures and N_CORES is the number of cpu cores to use for
multiprocessing (default is 1)

In order to generate the figure from appendix G use the following command:
python main --config example2 --dest_dir DEST_DIR --n_cores N_CORES

Notice that it takes a lot of time to run the simulation and the usage of multiple cpu cores is recommended.
You can change the parameter of the simulation by creating a new config in the config.py file and adding it to
the CONFIG dict variable.

Detailed explanations of the code will be shared through github if and when the paper is accepted.