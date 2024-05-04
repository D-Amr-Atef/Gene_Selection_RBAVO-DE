#
# ------------------------------------------------------------------------------------------------------%
# Created by "Ahmed Gad"                                                                                %
#                                                                                                       %
#       Email:      ahmed.gad_it@yahoo.com                                                              %
#       Homepage:   https://www.researchgate.net/profile/Ahmed_G_Gad                                    %
#       Github:     https://github.com/ahmedgad19891                                                    %
#-------------------------------------------------------------------------------------------------------%

from numpy import where
from pandas import DataFrame, MultiIndex

def bold_max(data):
        """
        highlight the maximum in a Series or DataFrame
        """
        attr = "font-weight: {}".format("bold")
        if data.ndim == 1:  # Series from .apply(axis=0) or axis=1
                is_max = data == data.max()
                return [attr if v else "" for v in is_max]
        else:  # from .apply(axis=None)
                is_max = data.groupby(level=1, axis=1).transform('max') == data
                return DataFrame(where(is_max, attr, ""),
                                    index=data.index, columns=data.columns)

def bold_min(data):
        """
        highlight the minimum in a Series or DataFrame
        """
        attr = "font-weight: {}".format("bold")
        if data.ndim == 1:  # Series from .apply(axis=0) or axis=1
                is_min = data == data.min()
                return [attr if v else "" for v in is_min]
        else:  # from .apply(axis=None)
                is_min = data.groupby(level=1, axis=1).transform('min') == data
                return DataFrame(where(is_min, attr, ""),
                                    index=data.index, columns=data.columns)

