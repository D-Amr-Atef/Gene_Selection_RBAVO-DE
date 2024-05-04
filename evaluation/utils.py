#
# ------------------------------------------------------------------------------------------------------%
# Created by "Ahmed Gad"                                                                                %
#                                                                                                       %
#       Email:      ahmed.gad_it@yahoo.com                                                              %
#       Homepage:   https://www.researchgate.net/profile/Ahmed_G_Gad                                    %
#       Github:     https://github.com/ahmedgad19891                                                    %
#-------------------------------------------------------------------------------------------------------%

from pandas import DataFrame, MultiIndex
from numpy import mean, std, sum, ceil, bincount
from datasets.dataset_splitter import DatasetSplitter
from os import getcwd, path, makedirs
from functions.style_function import *
import pickle as pkl 
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from copy import deepcopy
from models.root import Root

class Utils(Root):
        """ This is helper for the Evaluation """
        
        
        metrics = ["Fitness", "Accuracy", "Error_rate", "Feature_size", "Time", "Kappa", "Precision", "Recall", "F1_score"]
        terms = ["best", "mean", "worst", "std"]
        
        def __init__(self, dataset_list=None, algo_dicts=None, obj_fun_dicts=None, trans_fun_dicts=None):
                self.dataset_list = dataset_list
                self.algo_dicts = algo_dicts
                self.obj_fun_dicts = obj_fun_dicts
                self.trans_fun_dicts = trans_fun_dicts
                self.obj_funs, self.trans_funcs = self._list_funcs__()
                self.path_list = []
        
        def _list_funcs__(self):
                obj_funs = []
                trans_funcs = []
                
                for obj_func_cat, obj_func_list in self.obj_fun_dicts.items():
                    for ObjFunc in obj_func_list:
                            obj_funs = deepcopy(obj_funs + [ObjFunc.__name__])

                for trans_func_cat, trans_func_list in self.trans_fun_dicts.items():
                        for trans_func in trans_func_list:
                            trans_funcs = deepcopy(trans_funcs + [trans_func.__name__])
                return obj_funs, trans_funcs

        def _db_handler__(self):
                dataset_splits = {}
                for ds in self.dataset_list:
                        ds_splitter = DatasetSplitter(ds)
                        X_train, X_test, y_train, y_test = ds_splitter._split__(kfold=5)
                        dataset_splits[ds] = [X_train, X_test, y_train, y_test]
                return dataset_splits

        def _db_handler_Relief__(self):
                
                dataset_splits_Relief = {}
                for ds in self.dataset_list:
                        ds_splitter_Relief = DatasetSplitter(ds_name= ds, f_combina= self.f_combina)
                     
                        X_train_relief, X_test_relief, y_train_relief, y_test_relief = ds_splitter_Relief._split_Relief__(kfold=5)
                        dataset_splits_Relief[ds] = [X_train_relief, X_test_relief, y_train_relief, y_test_relief]
                return dataset_splits_Relief
            
        def _dir_maker__(self, ObjFunc, trans_func_cat):
                temp_path_list = []
                result_path = getcwd() + "/history"
                for metric in self.metrics:
                        output_path = path.join(result_path, ObjFunc.__name__, metric, trans_func_cat)
                        if not path.exists(output_path):
                                makedirs(output_path)
                        temp_path_list.append(output_path)
                self.path_list = deepcopy(temp_path_list)

        def _make_header__(self, *head):
                names = ["Datasets"]
                for _ in range(len(head)-1):
                        names.append(None)
                header = MultiIndex.from_product(head,
                                                 names=names)
                return header

        def _ax_creator__(self, nrows, idx, epoch, alph, ds):
                ax = plt.subplot(nrows, 3, idx+1)
                ax.yaxis.set_major_formatter(FormatStrFormatter("%.4f"))
                ax.set_xlim(left=0, right=epoch)
                ax.set(xlabel="Number of iterations", ylabel="Mean of fitness value")
                ax.set_title("({}) {}".format(alph,ds), y=-0.35)
                return ax
            
            
            

        def _sort_and_get_metrics(self, res=None, id_fitness=None, id_best=None, id_worst=None):
                sorted_res = sorted(res, key=lambda temp: (temp[id_fitness], sum(temp[self.ID_FEAT])))
                best_res = sorted_res[id_best]
                worst_res = sorted_res[id_worst]
                
                
                
                best_selected_features = best_res[self.ID_FEAT].astype(int)
                worst_selected_features = worst_res[self.ID_FEAT].astype(int)

                t_fit = [best_res[self.ID_FIT], mean(res[:,self.ID_FIT]),
                       worst_res[self.ID_FIT], std(res[:,self.ID_FIT])]
                t_acc = [best_res[self.ID_ACC], mean(res[:,self.ID_ACC]),
                       worst_res[self.ID_ACC], std(res[:,self.ID_ACC])]
                t_err = [best_res[self.ID_ERROR], mean(res[:,self.ID_ERROR]),
                       worst_res[self.ID_ERROR], std(res[:,self.ID_ERROR])]
                t_feat = [sum(best_selected_features), mean(list(map(sum, res[:,self.ID_FEAT]))),
                        sum(worst_selected_features), std(list(map(sum, res[:,self.ID_FEAT]))),
                        mean(list(map(sum, res[:,self.ID_FEAT])))/self.problem_size]
                t_time = [best_res[self.ID_TIME], mean(res[:,self.ID_TIME]),
                        worst_res[self.ID_TIME], std(res[:,self.ID_TIME])]
                
                t_kappa = [best_res[self.ID_kappa], mean(res[:,self.ID_kappa]),
                       worst_res[self.ID_kappa], std(res[:,self.ID_kappa])]
                t_precision = [best_res[self.ID_precision], mean(res[:,self.ID_precision]),
                       worst_res[self.ID_precision], std(res[:,self.ID_precision])]
                t_recall = [best_res[self.ID_recall], mean(res[:,self.ID_recall]),
                       worst_res[self.ID_recall], std(res[:,self.ID_recall])]
                t_f1_score = [best_res[self.ID_f1_score], mean(res[:,self.ID_f1_score]),
                       worst_res[self.ID_f1_score], std(res[:,self.ID_f1_score])]
                # t_specificity = [best_res[self.ID_specificity], mean(res[:,self.ID_specificity]),
                #        worst_res[self.ID_specificity], std(res[:,self.ID_specificity])]
                # t_CCI = [best_res[self.ID_CCI], mean(res[:,self.ID_CCI]),
                #        worst_res[self.ID_CCI], std(res[:,self.ID_CCI])]
                # t_ICI = [best_res[self.ID_ICI], mean(res[:,self.ID_ICI]),
                #        worst_res[self.ID_ICI], std(res[:,self.ID_ICI])]
                # t_ROC = [best_res[self.ID_ROC], mean(res[:,self.ID_ROC]),
                #        worst_res[self.ID_ROC], std(res[:,self.ID_ROC])]
                # t_MCC = [best_res[self.ID_MCC], mean(res[:,self.ID_MCC]),
                #        worst_res[self.ID_MCC], std(res[:,self.ID_MCC])]
                
                best_loss_train = best_res[self.ID_LOSS]
                
                
                # return [t_fit, t_acc, t_feat, t_time, best_loss_train], best_selected_features, worst_selected_features
                return [t_fit, t_acc, t_err, t_feat, t_time, best_loss_train, t_kappa, t_precision, t_recall, t_f1_score], best_selected_features, worst_selected_features


        def _df_styler__(self, df, metric, terms, head_len):
                styler = df.style
                for term in terms:
                        # if metric == "Accuracy" and term != "std":
                        if (metric == "Accuracy" or metric == "Kappa" or metric == "Precision" or metric == "Recall"  or metric == "F1_score") and term != "std":

                                df = styler.apply(bold_max, subset=df.columns.get_level_values(head_len)==term, axis=None)
                        else:
                                df = styler.apply(bold_min, subset=df.columns.get_level_values(head_len)==term, axis=None)
                return df

        def _output_results__(self, dfs, path_list, head, out_path):                
                for (df, func_path, metric) in zip(dfs, path_list, self.metrics):
                        terms = deepcopy(self.terms)
                        if metric == "Feature_size":
                                terms.append("selection ratio")
                        header = self._make_header__(*head, terms)
                        df = DataFrame.from_dict(df, columns=header, orient="index")
                        df.loc["Overall " + metric] = df.mean()
                        df = df.round(decimals=4)
##                        c = bincount(df.values.argmax(1), minlength=df.shape[1])
##                        print(metric, c)
                        df = self._df_styler__(df, metric, terms, len(head))
                        df.to_excel("{}/{}_{}_{}.xlsx".format(func_path, *out_path, metric))

        def _output_pkls_and_figs__(self, loss_dict, epoch, obj_func, trans_func):
                file = open("{}/{}_{}_{}".format(self.path_list[self.ID_FIT], obj_func, trans_func, "Fitness"), "wb")
                pkl.dump(loss_dict, file, pkl.HIGHEST_PROTOCOL)

                nrows = int(ceil(len(self.dataset_list)/3))
                figure = plt.figure(obj_func + "_" + trans_func, figsize=(3*6, nrows*4))
                for idx1, (ds, alph) in enumerate(zip(self.dataset_list, list("abcdefghijklmnopqrstuvwxyz"))):
                        ax = self._ax_creator__(nrows, idx1, epoch, alph, ds)
                        for idx2, (name, Algo) in enumerate(self.algo_dicts.items()):
                                ax.plot(loss_dict[ds][idx2], label=name)
                        ax.legend(loc="upper center", ncol=6, mode="expand")
                figure.tight_layout()
                figure.subplots_adjust(hspace=0.5, wspace=0.3)
                plt.savefig("{}/{}_{}_{}.pdf".format(self.path_list[self.ID_FIT], obj_func, trans_func, "Fitness"), dpi=150)
                
                
                
                

