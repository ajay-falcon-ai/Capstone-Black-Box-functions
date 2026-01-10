# flows/function1_data_handler.py
import os
from data.BaseDataHandler import BaseDataHandler

class Function5DataHandler(BaseDataHandler):
    
    def __init__(self):
        # Resolve paths relative to this file's folder
        current_dir = os.path.dirname(__file__)
        input_file = os.path.join(current_dir, "initial_inputs.npy") # initial inputs file
        output_file = os.path.join(current_dir, "initial_outputs.npy") # initial outputs file

        super().__init__(
            input_file=input_file, # initial inputs file
            output_file=output_file, # initial outputs file
            input_columns=["var-1", "var-2", "var-3", "var-4"], # schema for function 1
            output_label="yield" # output label for function 1
        )

    def add_weekly_updates(self):
        """
        Append weekly points specific to Function 5.
        Each week gets its own label (week-1, week-2, …).
        """
        self.append_week_point(1, [0.183674, 0.857143, 0.816327, 0.877551], 872.3074873002278)
        self.append_week_point(2, [0.272727, 0.848484, 0.848485, 0.818182], 743.1182481623213)
        self.append_week_point(3, [0.272727, 0.848484, 0.848485, 0.818182], 1118.8725469008125)
        self.append_week_point(4, [0.242085, 0.848724, 0.469207, 0.893286], 386.6818739904094)
        self.append_week_point(5, [0.263058, 0.962540, 0.979484, 1.057644], 4564.328857219503)
        self.append_week_point(6, [0.797599, 0.775484, 1.047075, 1.122867], 6619.923795891087)
        self.append_week_point(7, [0.353188, 0.520322, 1.112618, 1.186113], 6717.039990516725)
        self.append_week_point(8, [0.287950, 0.950000, 0.950000, 1.050000], 3942.207447141211)
        self.append_week_point(9, [0.803000, 0.790000, 1.070000, 1.140000], 7705.825094042719)
        self.append_week_point(10, [0.112614, 0.025111, 0.567469, 1.144827], 1246.5123230669396)
        self.append_week_point(11, [0.890679, 1.008486, 1.208429, 0.458510], 7827.226901543306)
        # Add more weeks as needed