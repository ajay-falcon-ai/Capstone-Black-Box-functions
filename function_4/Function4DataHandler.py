# flows/function1_data_handler.py
import os
from data.BaseDataHandler import BaseDataHandler

class Function4DataHandler(BaseDataHandler):
    
    def __init__(self):
        # Resolve paths relative to this file's folder
        current_dir = os.path.dirname(__file__)
        input_file = os.path.join(current_dir, "initial_inputs.npy") # initial inputs file
        output_file = os.path.join(current_dir, "initial_outputs.npy") # initial outputs file

        super().__init__(
            input_file=input_file, # initial inputs file
            output_file=output_file, # initial outputs file
            input_columns=["hype-para--1", "hype-para-2", "hype-para-3", "hype-para-4"], # schema for function 1
            output_label="opt-place-score" # output label for function 1
        )

    def add_weekly_updates(self):
        """
        Append weekly points specific to Function 4.
        Each week gets its own label (week-1, week-2, …).
        """
        self.append_week_point(1, [0.469388, 0.448980, 0.469388, 0.204082], -4.620232320523549)
        self.append_week_point(2, [0.515152, 0.484848, 0.363636, 0.181819], -4.939960051262183)
        self.append_week_point(3, [0.515152, 0.484848, 0.363636, 0.181819], -5.313374447160626)
        self.append_week_point(4, [0.111733, 0.041201, 0.540545, 0.286416], -12.867749036987885)
        self.append_week_point(5, [0.343612, 0.592249, 0.651353, 0.444513], -7.221570128386034)
        # Add more weeks as needed