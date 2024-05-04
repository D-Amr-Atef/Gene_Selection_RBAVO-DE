from numpy import arange, mean, flatnonzero
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import  confusion_matrix, precision_score, recall_score, f1_score, cohen_kappa_score,roc_curve
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt


class xgbTree:        
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
             
                
                self.fold = 5
                self.grid_search = False
                self.cross_validation = False
                
        def _fit__(self, solution=None, minmax=0):
                """
                    GridSearchCV works by training our model multiple times on a range of sepecified parameters.
                    That way, we can test our model with each parameter and figure out the optimal values to get the best accuracy results.
                """
                cols = flatnonzero(solution)
                max_depth=3
                eta=0.4
                gamma=0
                colsample_bytree=0.8
                min_child_weight=1 
                subsample=0.75
                train_data = self.X_train[:,cols]
                test_data = self.X_test[:,cols]
                
                
    
  
     
                # k = 5
                if self.grid_search:
                        # knn_gs = KNeighborsClassifier()  
                        XGB_gs = XGBClassifier()                               
                        param_grid = {"n_neighbors": arange(1, 10)}             #create a dictionary of all values we want to test for n_neighbors
                        knn_gscv = GridSearchCV(XGB_gs, param_grid, cv=self.kfold)   #use gridsearch to test all values for n_neighbors
                        knn_gscv.fit(train_data, self.y_train)                               #fit model to data
                        # k = knn_gscv.best_params_["n_neighbors"]                #check top performing n_neighbors value
                
                clf = XGBClassifier(max_depth=max_depth, eta=eta, gamma=gamma, colsample_bytree=colsample_bytree, min_child_weight=min_child_weight , subsample=subsample)    

               
                
                
                clf.fit(train_data, self.y_train)
                
                # print(y_pred)
                # predictions = [round(value) for value in y_pred]
               
                if self.cross_validation:
                        cv_scores = cross_val_score(clf, train_data, self.y_train, cv=self.kfold)
 
                        err = [1 - x for x in cv_scores]
                else:
                        score = clf.score(test_data, self.y_test)
                        # accuracy = accuracy_score(self.y_test, y_pred)

                        err = 1 - score
                
                # return mean(err)               
                
                ##################################################################
                # print("Accuracy: %.2f%%" % (accuracy * 100.0))
                # confusion_matrix(self.y_test,y_pred)
                
                # drawing confusion matrix
                # pd.crosstab(self.y_test, y_pred, rownames = ['Actual'], colnames =['Predicted'], margins = True)
                ##################################################################
                y_pred = clf.predict(test_data)
                # y_pred_prob = clf.predict_proba(test_data)
                
                F1_score_res = f1_score(self.y_test, y_pred, average='binary')
                Precision_res = precision_score(self.y_test, y_pred, average='binary')
                Recall_res = recall_score(self.y_test, y_pred, average='binary')
                kappa_res = cohen_kappa_score(self.y_test, y_pred)
                
                # cm1 = confusion_matrix(self.y_test,y_pred)
                # Specificity = cm1[0,0]/(cm1[0,0]+cm1[0,1])
                # Sensitivity = cm1[1,1]/(cm1[1,0]+cm1[1,1])
                
                
                
                # false_positive_rate1, true_positive_rate1, threshold1 = roc_curve(self.y_test, y_pred_prob)
                
                # plt.subplots(1, figsize=(10,10))
                # plt.title('The ROC curve of the proposed IBAVO_AO algorithm')
                # plt.plot(false_positive_rate1, true_positive_rate1)
                # plt.plot([0, 1], ls="--")
                # plt.plot([0, 0], [1, 0] , c=".7"), plt.plot([1, 1] , c=".7")
                # plt.ylabel('Sensitivity')
                # plt.xlabel('Specificity')
                # plt.show()

    

                # fpr_res, tpr_res, _ = roc_curve(self.y_test, y_pred)
                ##################################################################
                # print(Specificity, Sensitivity)
                return err, kappa_res, Precision_res, Recall_res, F1_score_res

                # return mean(err), mean(F1_score_res), mean(Precision_res), mean(Recall_res)