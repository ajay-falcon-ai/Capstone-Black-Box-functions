# flows/function1_data_handler.py
import os
from data.BaseDataHandler import BaseDataHandler

class Function7DataHandler(BaseDataHandler):
    
    def __init__(self):
        # Resolve paths relative to this file's folder
        current_dir = os.path.dirname(__file__)
        input_file = os.path.join(current_dir, "initial_inputs.npy") # initial inputs file
        output_file = os.path.join(current_dir, "initial_outputs.npy") # initial outputs file

        super().__init__(
            input_file=input_file, # initial inputs file
            output_file=output_file, # initial outputs file
            input_columns=["hype-para-1", "hype-para-2", "hype-para-3", "hype-para-4", "hype-para-5", "hype-para-6"], # schema for function 1
            output_label="score" # output label for function 1
        )

    def add_weekly_updates(self):
        """
        Append weekly points specific to Function 7.
        Each week gets its own label (week-1, week-2, …).
        """
        self.append_week_point(1, [0.000000, 0.500000, 0.214286, 0.214286, 0.428572, 0.714286], 1.181422704250112)
        self.append_week_point(2, [0.060606, 0.484848, 0.242424, 0.181818, 0.484848, 0.727273], 0.9769171261966855)
        self.append_week_point(3, [0.038479, 0.518839, 0.311248, 0.237261, 0.417124, 0.684376], 1.415182353706698)
        self.append_week_point(4, [0.107718, 0.586286, 0.175377, 0.303162, 0.560604, 0.684376], 0.4790359561505474)
        self.append_week_point(5, [0.000000, 0.990970, 0.990603, 0.000000, 0.000000, 1.017683], 0.028769356636723045)
        self.append_week_point(6, [0.090408, 0.599655, 0.299291, 0.287006, 0.599215, 0.826037], 0.33553729563585927)
        self.append_week_point(7, [0.056006, 0.252690, 0.914472, 0.121294, 0.362425, 1.007224], 0.9161953953665816)
        self.append_week_point(8, [0.067812, 0.504863, 0.287394, 0.230127, 0.473182, 0.728104], 1.1241859989248546)
        self.append_week_point(9, [0.049863, 0.526174, 0.301564, 0.217309, 0.491237, 0.791235], 0.8608279363272374)
        self.append_week_point(10, [0.170167, 0.032372, 0.330954, 0.272601, 0.031284, 1.076438], 0.2237511524097176)
        # Add more weeks as needed