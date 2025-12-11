# flows/function1_data_handler.py
import os
from data.BaseDataHandler import BaseDataHandler

class Function3DataHandler(BaseDataHandler):
    
    def __init__(self):
        # Resolve paths relative to this file's folder
        current_dir = os.path.dirname(__file__)
        input_file = os.path.join(current_dir, "initial_inputs.npy") # initial inputs file
        output_file = os.path.join(current_dir, "initial_outputs.npy") # initial outputs file

        super().__init__(
            input_file=input_file, # initial inputs file
            output_file=output_file, # initial outputs file
            input_columns=["compound-1", "compound-2", "compound-3"], # schema for function 1
            output_label="trans-side-effect" # output label for function 1
        )

    def add_weekly_updates(self):
        """
        Append weekly points specific to Function 3.
        Each week gets its own label (week-1, week-2, …).
        """
        self.append_week_point(1, [0.464647, 0.666667, 0.171718], -0.11118571432117087)
        self.append_week_point(2, [0.303031, 0.242424, 0.474748], -0.030680318237259866)
        self.append_week_point(3, [0.263346, 0.138532, 0.363743], -0.048179209676502605)
        self.append_week_point(4, [0.448900, 0.470829, 0.414315], -0.02040441290319508)
        self.append_week_point(5, [0.357938, 0.038532, 0.517008], -0.1175388643248932)
        # Add more weeks as needed