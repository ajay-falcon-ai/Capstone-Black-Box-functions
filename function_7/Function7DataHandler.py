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
        #self.append_week_point(1, [0.000000, 0.500000, 0.214286, 0.214286, 0.428572, 0.714286], 1.181422704250112)
        #self.append_week_point(2, [0.060606, 0.484848, 0.242424, 0.181818, 0.484848, 0.727273], 0.9769171261966855)
        #self.append_week_point(3, [0.038479, 0.518839, 0.311248, 0.237261, 0.417124, 0.684376], 1.415182353706698)
        #self.append_week_point(4, [0.107718, 0.586286, 0.175377, 0.303162, 0.560604, 0.684376], 0.4790359561505474)
        #self.append_week_point(5, [0.000000, 0.990970, 0.990603, 0.000000, 0.000000, 1.017683], 0.028769356636723045)
        # Add more weeks as needed