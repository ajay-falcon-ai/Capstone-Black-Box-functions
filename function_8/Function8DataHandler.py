# flows/function1_data_handler.py
import os
from data.BaseDataHandler import BaseDataHandler

class Function8DataHandler(BaseDataHandler):
    
    def __init__(self):
        # Resolve paths relative to this file's folder
        current_dir = os.path.dirname(__file__)
        input_file = os.path.join(current_dir, "initial_inputs.npy") # initial inputs file
        output_file = os.path.join(current_dir, "initial_outputs.npy") # initial outputs file

        super().__init__(
            input_file=input_file, # initial inputs file
            output_file=output_file, # initial outputs file
            input_columns=["hype-para-1", "hype-para-2", "hype-para-3", "hype-para-4", "hype-para-5", "hype-para-6", "hype-para-7", "hype-para-8"], # schema for function 1
            output_label="output" # output label for function 1
        )

    def add_weekly_updates(self):
        """
        Append weekly points specific to Function 8.
        Each week gets its own label (week-1, week-2, …).
        """
        self.append_week_point(1, [0.000000, 0.125000, 0.000000, 0.000000, 0.375000, 0.750000, 0.500000, 0.875000], 9.5658)
        self.append_week_point(2, [0.121212, 0.121212, 0.000000, 0.000000, 0.363636, 0.727273, 0.484849, 0.848484], 9.6097592309634)
        self.append_week_point(3, [0.043751, 0.045306, 0.045319, 0.033695, 0.337741, 0.771663, 0.496881, 0.775992], 9.5876553462161)
        self.append_week_point(4, [0.480871, 0.190087, 0.193605, 0.710357, 0.197956, 0.208186, 0.219257, 0.503654], 9.1140770207914)
        self.append_week_point(5, [0.000000, 0.000000, 0.000000, 0.845416, 0.918723, 0.589616, 0.000000, 0.000000], 9.2921179841235)
        self.append_week_point(6, [0.010000, 0.661416, 0.010000, 0.212650, 0.889104, 0.671374, 0.010000, 0.450308], 9.5673490736736)
        self.append_week_point(7, [0.138474, 0.190151, 0.378834, 0.303465, 1.057490, 0.950823, 0.074575, 0.795668], 9.5144374297026)
        self.append_week_point(8, [0.045306, 0.045319, 0.045319, 0.033695, 0.337741, 0.771663, 0.496881, 0.775992], 9.5880031008211)
        self.append_week_point(9, [0.150000, 0.150000, 0.050000, 0.050000, 0.350000, 0.750000, 0.500000, 0.850000], 9.6158)
        # Add more weeks as needed