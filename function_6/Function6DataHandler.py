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
        self.append_week_point(6, [0.697910, 0.990583, 1.035622, 0.974136, 0.655285], -2.0390133791009304)
        self.append_week_point(7, [0.909574, 1.046923, 0.862744, 0.923652, 0.853835], -2.3063646975145775)
        self.append_week_point(8, [0.650000, 0.200000, 0.700000, 0.700000, 0.150000], -0.6349274195629477)
        self.append_week_point(9, [0.700000, 0.150000, 0.750000, 0.700000, 0.100000], -0.7453415482672865)
        self.append_week_point(10, [0.660796, 1.092932, 0.153903, 0.052573, 0.959181], -3.039360793240201)
        self.append_week_point(11, [1.044814, 1.182954, 0.060528, 0.012709, 1.023245], -3.778114188571625)
        # Add more weeks as needed