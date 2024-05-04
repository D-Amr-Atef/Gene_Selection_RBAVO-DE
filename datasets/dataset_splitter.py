#
# ------------------------------------------------------------------------------------------------------%
# Created by "Ahmed Gad"                                                                                %
#                                                                                                       %
#       Email:      ahmed.gad_it@yahoo.com                                                              %
#       Homepage:   https://www.researchgate.net/profile/Ahmed_G_Gad                                    %
#       Github:     https://github.com/ahmedgad19891                                                    %
#-------------------------------------------------------------------------------------------------------%

from pandas import read_csv
from sklearn.model_selection import train_test_split
from datasets.sklearn_relief import Relief
from os import getcwd, path
from numpy import ones,shape
from models.root import Root

class DatasetSplitter(Root):
        def __init__(self, ds_name=None,trans_func=None, f_combina=None):
                self.trans_func=trans_func
                PATH_DATASETS = "datasets/RNA/csvRNA/"

                dataset_path = PATH_DATASETS + ds_name + ".csv"
                df = read_csv(dataset_path, sep=",", nrows=None)
                
                self.feat = df.values[:,:-1]
                self.label = df.values[:,-1]
              
                #################################################################################
                ################## RELIEF Algorithm ##################
                r = Relief(n_features = f_combina, n_jobs = 1, random_state = 42)
               
                my_transformed_matrix = r.fit_transform(self.feat,self.label)
                self.feat_relief = my_transformed_matrix
                self.label_relief = df.values[:,-1]
              
                #################################################################################
                #################################################################################
  

#################################################################################
#################################################################################
        def _split__(self, kfold=5):
                
                test_size = 1 / kfold
                X_train, X_test, y_train, y_test = train_test_split(self.feat, self.label, test_size=test_size, random_state=42)
                return X_train, X_test, y_train, y_test

#################################################################################
#################################################################################        
        def _split_Relief__(self, kfold=5):
                
                test_size = 1 / kfold
                X_train_relief, X_test_relief, y_train_relief, y_test_relief = train_test_split(self.feat_relief, self.label_relief, test_size=test_size, random_state=42)
                return X_train_relief, X_test_relief, y_train_relief, y_test_relief
