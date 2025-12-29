# flows/function1_data_handler.py
import os
from data.BaseDataHandler import BaseDataHandler

class Function1DataHandler(BaseDataHandler):
    
    def __init__(self):
        # Resolve paths relative to this file's folder
        current_dir = os.path.dirname(__file__)
        input_file = os.path.join(current_dir, "initial_inputs.npy") # initial inputs file
        output_file = os.path.join(current_dir, "initial_outputs.npy") # initial outputs file

        super().__init__(
            input_file=input_file, # initial inputs file
            output_file=output_file, # initial outputs file
            input_columns=["cont-src-1", "cont-src-2"], # schema for function 1
            output_label="radiation-reading" # output label for function 1
        )

    def add_weekly_updates(self):
        """
        Append weekly points specific to Function 1.
        Each week gets its own label (week-1, week-2, …).
        """
        self.append_week_point(1, [0.848485, 0.777778], -1.2487829338434956e-49)
        self.append_week_point(2, [0.404040, 0.959596], -9.343447791332904e-112)
        self.append_week_point(3, [0.579291, 1.059596], 4.5074380138539914e-133)
        self.append_week_point(4, [0.436769, 0.825163], -1.5199719684953676e-55)
        self.append_week_point(5, [0.469395, 1.050091], -6.688485279296031e-142)
        self.append_week_point(6, [0.472352, 1.055087], -1.8022788966239472e-144)
        self.append_week_point(7, [0.523861, 1.067166], -1.3151574669815363e-142)
        self.append_week_point(8, [0.650000, 0.680000], -0.004236010437020722)
        self.append_week_point(9, [0.650000, 0.690000], -0.0011513293804727184)
        # Add more weeks as needed