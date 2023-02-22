The code can be run as requested by running 'main.py' from terminal. It has two arguments:
- --excel-file
this should be a string value referring to the path of the validation or test dataset (in xlsx format).
- --baseline
this argument will measure either the baseline or the model performance. If a string value of 'baseline' is given, then the baseline will be run instead of the model.

The model requires a TestEnv.py file in order for it to work, as well as the q_agent.py and environment.py (provided in this directory) for the pickled Q Agent (q_agent.pickle) to be properly loaded.

Furthermore, the files provided show how our agent was trained, evaluated, and the performance visualized (e.g. when the model buys and sells on a price graph. For this, use the visualize.py file and modify the last line of code to refer to an evaluation csv file). The files for training and validation are also included as well as the notebook that prepares our data. We have not included a baseline model in this directory because it's behavior is essentially captured by the few lines of relevant code in the main.py file.

To train our model (the appropriate seed and hyperparameters are provided), simply set the value of eval to False (line 284), run the script and wait a few hours.
To evaluate the most recent model, just set the value of eval to True. Thsi will create an output csv.file which can be given a name in the last line of the code
Enjoy!
