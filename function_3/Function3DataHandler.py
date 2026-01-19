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
        self.append_week_point(6, [0.116130, 0.051462, 0.314168], -0.1482910205289665)
        self.append_week_point(7, [0.137356, 0.196578, 0.292441], -0.12635616438454833)
        self.append_week_point(8, [0.394576, 0.615328, 0.367214], -0.037692123265023345)
        self.append_week_point(9, [0.450000, 0.470000, 0.400000], -0.029031619136784075)
        self.append_week_point(10, [0.476973, 1.036177, 1.085450], -0.751389275062385)
        self.append_week_point(11, [1.063348, 1.133355, 0.010000], -0.1383251424785816)
        self.append_week_point(12, [1.163162, 1.215793, 1.184747], -0.45395156724018193)
        # Add more weeks as needed