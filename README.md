# constraintKG
Objective funtion is written in funciton obj_func() in cKG.py.

Running experiment in debug mode:
Execute the function debug_main() in main.py.

Runing multiple experiments:
below "if __name__ == '__main__':"
	comment the line debug_main() in main.py 
	uncomment the function main() function in main.py
In terminal, enter the command main.py <folder to save data and log>

Continuous experiment: if you already run the experiment and have the directory, then when the program restart, it will read the .pkl file and continue.
*Note: if you run a experiment but not have .pkl file store, you need to delete the experiment folder*



