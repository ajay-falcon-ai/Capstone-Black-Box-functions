import numpy as np
import pandas as pd

class BaseDataHandler:
    def __init__(self, input_file, output_file, input_columns, output_label):
        # Load initial data
        self.inputs = np.load(input_file, allow_pickle=True)
        self.outputs = np.load(output_file, allow_pickle=True)
        self.input_columns = input_columns
        self.output_label = output_label
        self.source_labels = ['file'] * len(self.inputs)
        self.df_sorted = None

    def build_dataframe(self):
        df = pd.DataFrame(self.inputs, columns=self.input_columns)
        df[self.output_label] = self.outputs
        df['source'] = self.source_labels
        self.df_sorted = df.sort_values(by=self.output_label, ascending=False)
        return self.df_sorted

    def append_week_point(self, week_number, input_point, output_value):
        """Add a single weekly (input, output) point with source label 'week-x'."""
        self.inputs = np.append(self.inputs, [input_point], axis=0)
        self.outputs = np.append(self.outputs, [output_value], axis=0)
        self.source_labels.append(f"week-{week_number}")

    def add_candidates(self, candidates, predicted_y):
        """
        Add candidate points from optimisation results.
        Each candidate gets source label 'week-x'.
        """
        for x, y in zip(candidates, predicted_y):
            self.inputs = np.append(self.inputs, [x], axis=0)
            self.outputs = np.append(self.outputs, [y], axis=0)
            self.source_labels.append("week-x")   # fixed label

    def extract_candidate_arrays(self, query_results):
        """
        Return numpy arrays: candidates (K, d) and predicted_y (K,)
        from logged query_results. Uses 'y_pred' if present, else falls back to 'score'.
        """
        import numpy as np
        candidates = np.array([entry['x'] for entry in query_results])
        if query_results and ('y_pred' in query_results[0]):
            predicted_y = np.array([entry['y_pred'] for entry in query_results])
        else:
            predicted_y = np.array([entry['score'] for entry in query_results])
        return candidates, predicted_y

    def append_candidates_to_df_copy(
            self,
            candidates,
            predicted_y=None,
            source_label="week-x",
            feature_cols=None,
            yield_col=None,
            source_col="source"
        ):
            """
            Return (df_copy, new_indices) where df_copy is a copy of the handler's dataframe
            with candidate rows appended. Does NOT mutate handler state.

            - candidates: np.ndarray shape (K, d) or (K,)
            - predicted_y: np.ndarray shape (K,) or None (will be NaN if None)
            - feature_cols: optional list of column names to map features into
            - yield_col: optional name of the yield/output column
            - source_col: name of the provenance column (created if missing)
            """
            import numpy as np
            import pandas as pd

            # Build base dataframe copy
            df = self.build_dataframe().copy().reset_index(drop=True)

            # Normalize candidates to 2D
            cand = np.asarray(candidates)
            if cand.ndim == 1:
                cand = cand.reshape(-1, 1)
            K, d = cand.shape

            # Prepare predicted_y
            if predicted_y is None:
                pred = np.array([np.nan] * K)
            else:
                pred = np.asarray(predicted_y).ravel()
                if len(pred) != K:
                    raise ValueError("predicted_y length must match number of candidates")

            # Determine feature column names
            if feature_cols is not None:
                if len(feature_cols) != d:
                    raise ValueError("feature_cols length must match candidate feature dimension")
                feat_cols = list(feature_cols)
            else:
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                if len(numeric_cols) >= d:
                    feat_cols = numeric_cols[:d]
                else:
                    feat_cols = numeric_cols + [f"x{i}" for i in range(len(numeric_cols), d)]

            # Ensure feature columns exist
            for col in feat_cols:
                if col not in df.columns:
                    df[col] = np.nan

            # Determine yield column heuristically if not provided
            if yield_col is not None and yield_col in df.columns:
                ycol = yield_col
            else:
                candidates_names = [c for c in df.columns if any(k in c.lower() for k in ("yield", "radiation", "output", "y", "score"))]
                if candidates_names:
                    ycol = candidates_names[0]
                else:
                    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                    if numeric_cols:
                        ycol = numeric_cols[-1]
                    else:
                        ycol = "y"
                        if ycol not in df.columns:
                            df[ycol] = np.nan

            # Ensure source column exists
            if source_col not in df.columns:
                df[source_col] = ""

            # Build new rows dict
            new_rows = {col: [np.nan] * K for col in df.columns}
            for j, col in enumerate(feat_cols):
                new_rows[col] = cand[:, j].tolist()
            new_rows[ycol] = pred.tolist()
            new_rows[source_col] = [source_label] * K

            new_df = pd.DataFrame(new_rows, columns=df.columns)
            df_copy = pd.concat([df, new_df], ignore_index=True)

            start_idx = len(df_copy) - K
            new_indices = list(range(start_idx, start_idx + K))
            return df_copy, new_indices


    def sort_by_penultimate(self, df=None, ascending=False, reset_index=True):
        """
        Sort df (or the handler's built dataframe if df is None) by the penultimate column.
        Returns a new DataFrame.
        """
        import pandas as pd
        if df is None:
            df = self.build_dataframe()
        col = df.columns[-2]
        df_sorted = df.sort_values(by=col, ascending=ascending, na_position='last')
        if reset_index:
            df_sorted = df_sorted.reset_index(drop=True)
        return df_sorted
    
    def get_inputs(self):
        return self.inputs

    def get_outputs(self):
        return self.outputs