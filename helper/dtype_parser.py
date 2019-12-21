import numpy as np
import pandas as pd

class DtypeParse(object):
    def __init__(self,df):
        self.df = df

    def parse_sparsity(self):
        cols = self.df.columns
        out_dict = {}
        for col in cols:
            if self.df[col].dtype == np.float64 or self.df[col].dtype == np.int64:
                length = self.df[col].shape[0]
                idx = self.df[col] == 0
                zeros = idx.sum()

                ratio = zeros / length
                out_dict[col] = ratio

        return pd.Series(out_dict).sort_values()

    def parse_bool(self):
        cols = self.df.columns
        out_dict = {}
        for col in cols:
            if self.df[col].dtype == np.bool:
                length = self.df[col].shape[0]
                idx = self.df[col] == 0
                zeros = idx.sum()

                ratio = zeros / length
                out_dict[col] = ratio

        return pd.Series(out_dict)

    def parse_objects(self):
        cols = self.df.columns
        out_dict = {}
        for col in cols:
            if self.df[col].dtype == np.bool or self.df[col].dtype == np.float64 or self.df[col].dtype == np.int64:
                pass
            else:
                out_dict[col] = self.df[col].dtype
        return pd.Series(out_dict)

    def draw_hist(self,col, keep):
        threshold = self.df[col].quantile(keep)
        col[col < threshold].hist(bins=100)
