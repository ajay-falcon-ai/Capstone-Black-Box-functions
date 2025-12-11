# flows/function1_data_handler.py
import os
from data.BaseDataHandler import BaseDataHandler

class Function6DataHandler(BaseDataHandler):
    
    def __init__(self):
        # Resolve paths relative to this file's folder
        current_dir = os.path.dirname(__file__)
        input_file = os.path.join(current_dir, "initial_inputs.npy") # initial inputs file
        output_file = os.path.join(current_dir, "initial_outputs.npy") # initial outputs file

        super().__init__(
            input_file=input_file, # initial inputs file
            output_file=output_file, # initial outputs file
            input_columns=["floor", "sugar", "eggs", "butter", "milk"], # schema for function 1
            output_label="flavour-score" # output label for function 1
        )

    def add_weekly_updates(self):
        """
        Append weekly points specific to Function 6.
        Each week gets its own label (week-1, week-2, …).
        """
        self.append_week_point(1, [0.733334, 0.200000, 0.666667, 0.733334, 0.133334], -0.5377942602933444)
        self.append_week_point(2, [0.757576, 0.101010, 0.707071, 0.757576, 0.151515], -0.7102415173770004)
        self.append_week_point(3, [0.667597, 0.157201, 0.738339, 0.678368, 0.179636], -0.6833253974004205)
        self.append_week_point(4, [0.578220, 0.311822, 0.237757, 0.422315, 0.588271], -1.4991274726694743)
        self.append_week_point(5, [0.501329, 0.211390, 0.390924, 0.970550, 0.015922], -0.8866184075971868)
        # Add more weeks as needed