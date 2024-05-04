
from numpy import arange, mean, shape, flatnonzero
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import  confusion_matrix, precision_score, recall_score, f1_score, cohen_kappa_score

class DT:        
        # def __init__(self, X_train, X_test, y_train, y_test):
        def __init__(self, x_train, x_test, y_train, y_test):
                # self.X_train = X_train
                # self.X_test = X_test
                # self.y_train = y_train
                # self.y_test = y_test
                self.X_train = x_train
                self.X_test = x_test
                self.y_train = y_train
                self.y_test = y_test
              
                
                self.kfold = 5
                self.grid_search = False
                self.cross_validation = False
                
        def _fit__(self, solution=None, minmax=0):
                """
                    GridSearchCV works by training our model multiple times on a range of sepecified parameters.
                    That way, we can test our model with each parameter and figure out the optimal values to get the best accuracy results.
                """
                cols = flatnonzero(solution)

                criterion = "gini"
                max_depth = 5
                max_features = 1
                random_state = 42
                
                if self.grid_search:
                        tree_gs = DecisionTreeClassifier()                             #create a dictionary of all values we want to test for c, kernel, degree, gamma
                        param_grid = {"n_estimators":arange(10,100,10),
                                      "criterion":["gini","entropy"],
                                      "max_depth":arange(1,10),
                                      "max_features": arange(1,shape(self.X_train)[1]),
                                      "random_state": [0,1]}        
                        tree_gscv = GridSearchCV(tree_gs, param_grid, cv=self.kfold)        #use gridsearch to test all values for criterion, max_depth
                        tree_gscv.fit(feat, label)                                        #fit model to data
                        criterion = tree_gscv.best_params_["criterion"]                   #check top performing criterion
                        max_depth = tree_gscv.best_params_["max_depth"]                   #check top performing max_depth value
                        max_features = tree_gscv.best_params_["max_features"]             #check top performing max_features value
                        random_state = tree_gscv.best_params_["random_state"]             #check top performing random_state value
                
                clf = DecisionTreeClassifier(criterion=criterion, max_depth=max_depth, max_features=max_features, random_state=random_state)

                train_data = self.X_train[:,cols]
                test_data = self.X_test[:,cols]
                
                clf.fit(train_data, self.y_train)
                
                if self.cross_validation:
                        cv_scores = cross_val_score(clf, self.feat[:,cols], label, cv=self.kfold)
                        err = [1 - x for x in cv_scores]
                else:
                        score = clf.score(test_data, self.y_test)
                        err = 1 - score
                
                
              
                        
                # return mean(err)
                ##################################################################
                y_pred = clf.predict(test_data)
                # confusion_matrix(self.y_test,y_pred)
                
                # drawing confusion matrix
                # pd.crosstab(self.y_test, y_pred, rownames = ['Actual'], colnames =['Predicted'], margins = True)
                ##################################################################
                
                F1_score_res = f1_score(self.y_test, y_pred, average='binary')
                Precision_res = precision_score(self.y_test, y_pred, average='binary')
                Recall_res = recall_score(self.y_test, y_pred, average='binary')
                kappa_res = cohen_kappa_score(self.y_test, y_pred)
                ##################################################################
                # print(F1_score_res, Precision_res, Recall_res )
                return err, kappa_res, Precision_res, Recall_res, F1_score_res

                # return mean(err), mean(F1_score_res), mean(Precision_res), mean(Recall_res)