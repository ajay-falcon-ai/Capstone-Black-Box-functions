# flows/function1_data_handler.py
import os
from data.BaseDataHandler import BaseDataHandler

class Function2DataHandler(BaseDataHandler):
    
    def __init__(self):
        # Resolve paths relative to this file's folder
        current_dir = os.path.dirname(__file__)
        input_file = os.path.join(current_dir, "initial_inputs.npy") # initial inputs file
        output_file = os.path.join(current_dir, "initial_outputs.npy") # initial outputs file

        super().__init__(
            input_file=input_file, # initial inputs file
            output_file=output_file, # initial outputs file
            input_columns=["number-1", "number-2"], # schema for function 1
            output_label="log-likelyhood-score" # output label for function 1
        )

    def add_weekly_updates(self):
        """
        Append weekly points specific to Function 1.
        Each week gets its own label (week-1, week-2, …).
        """
        self.append_week_point(1, [0.818182, 0.939394], 0.051646432712366415)
        self.append_week_point(2, [0.616162, 1.000000], 0.18011924802681153)
        self.append_week_point(3, [0.722766, 0.839711], 0.5338269840060812)
        self.append_week_point(4, [0.691873, 0.263356], 0.5960813635984847)
        self.append_week_point(5, [0.790304, 0.104833], 0.01686172630224915)
        self.append_week_point(6, [0.977791, 0.261960], 0.021320835230587712)
        self.append_week_point(7, [1.077791, 0.617990], 0.04170782561126679)
        self.append_week_point(8, [0.715000, 0.950000], 0.5976293971773099)
        # Add more weeks as needed