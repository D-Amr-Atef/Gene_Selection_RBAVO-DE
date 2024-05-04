#
# ------------------------------------------------------------------------------------------------------%
# Created by "Ahmed Gad"                                                                                %
#                                                                                                       %
#       Email:      ahmed.gad_it@yahoo.com                                                              %
#       Homepage:   https://www.researchgate.net/profile/Ahmed_G_Gad                                    %
#       Github:     https://github.com/ahmedgad19891                                                    %
#-------------------------------------------------------------------------------------------------------%

from numpy import shape, array, zeros, ones, ceil
from copy import deepcopy
from pandas import DataFrame
from time import time
from os import getcwd, path, makedirs
from evaluation.utils import Utils

class Evaluation(Utils):

        ID_FIT = 0      # best fintness
        ID_ACC = 1      # best accuracy
        ID_ERROR = 2    # error rate
        ID_FEAT = 3     # best selected features
        ID_TIME = 4     # best processing time
        ID_LOSS = 5     # best loss train      
        
        ID_kappa = 6
        ID_precision = 7
        ID_recall = 8 
        ID_f1_score = 9
        # ID_specificity = 10
        # ID_CCI = 11
        # ID_ICI = 12
        # ID_ROC = 13
        # ID_MCC = 14

        def __init__(self, dataset_list=None, algo_dicts=None, obj_fun_dicts=None, trans_fun_dicts=None, num_runs=30, domain_range=(0,1), log=True,
                     epoch=100, pop_size=30, lsa_epoch=10, f_combina=None):
            
                Utils.__init__(self, dataset_list, algo_dicts, obj_fun_dicts, trans_fun_dicts)
                self.num_runs = num_runs
                self.domain_range = domain_range
                self.log = log
                self.epoch = epoch
                self.pop_size = pop_size
                self.lsa_epoch = lsa_epoch
                self.f_combina = f_combina
                
        def _eval__(self):
                dataset_splits = self._db_handler_Relief__()
               
                
                if len(self.algo_dicts) == 1:
                        fit_dict, acc_dict, err_dict, feat_dict, time_dict, loss_dict, kappa_dict, precision_dict, recall_dict , f1_score_dict = ( dict.fromkeys(self.dataset_list , []) for _ in range(len(self.metrics)+1) )
                
                for obj_fun_cat, obj_fun_list in self.obj_fun_dicts.items():
                        
                        for ObjFunc in obj_fun_list:
                                
                                for trans_func_cat, trans_func_list in self.trans_fun_dicts.items():
                                        if len(self.algo_dicts) == 1:
                                                result_path = getcwd() + "/history"
                                                if not path.exists(result_path):
                                                        makedirs(result_path)
                                        else:
                                                self._dir_maker__(ObjFunc, trans_func_cat)
                                        
                                        for trans_func in trans_func_list:
                                                temp_fit_dict, temp_acc_dict, temp_err_dict, temp_feat_dict, temp_time_dict, temp_loss_dict, temp_kappa_dict, temp_precision_dict, temp_recall_dict , temp_f1_score_dict = ( dict.fromkeys(self.dataset_list , []) for _ in range(len(self.metrics)+1) )
                                                
                                                for ds in self.dataset_list:
                                                        fit_list, acc_list, err_list, feat_list, time_list, loss_list, kappa_list, precision_list, recall_list, f1_score_list = ( [] for _ in range(len(self.metrics)+1) )                                                               

                                                        self.problem_size = shape(dataset_splits[ds][0])[1]
                                                        
                                                        
                                                        ml = ObjFunc(*dataset_splits[ds])
                                                        before_err, before_kappa, before_precision, before_recall, before_f1_score = ml._fit__(ones(self.problem_size))
                                                        before_fit = self.OMEGA * before_err + (1 - self.OMEGA) * (sum(ones(self.problem_size)) / self.problem_size)
                                                        before_accuracy = (1 - before_err) 
                                                        
                                                        ##################################################################################
                                                        ##################################################################################
                                                        ##################################################################################
                                                        
                                                        print("> Dataset: {}, Objective Function: {} ==> {}, Transfer Function: {} ==> {}, Dataset Shape (Samples ==> {}, Features ==> {}), Training Dataset (Samples ==> {}, Features ==> {}), Testing Dataset (Samples ==> {}, Features ==> {})"
                                                              .format(ds, obj_fun_cat, ObjFunc.__name__, trans_func_cat, trans_func.__name__, shape(dataset_splits[ds][0])[0] + shape(dataset_splits[ds][1])[0] , shape(dataset_splits[ds][0])[1], shape(dataset_splits[ds][0])[0], shape(dataset_splits[ds][0])[1], shape(dataset_splits[ds][1])[0], shape(dataset_splits[ds][1])[1]))
                                                           
                                                        
                                                        
                                                        print("-------------------------------------------------------------------------")
                                                        # print("^^^Before Feature Selection:")
                                                        print("^^^Accuracy of Original Dataset (Before Feature Selection) ( {} ) : {:.4f}".format(ObjFunc.__name__, before_accuracy))
                                                        print("^^^Error rate of Original Dataset (Before Feature Selection) ( {} ) : {:.4f}".format(ObjFunc.__name__, before_err))                                                        
                                                        print("^^^Fitness of Original Dataset (Before Feature Selection) ( {} ) : {:.4f}".format(ObjFunc.__name__, before_fit))                                                        
                                                        
                                                        print("^^^Kappa of Original Dataset (Before Feature Selection) ( {} ) : {:.4f}".format(ObjFunc.__name__, before_kappa))
                                                        print("^^^Precision of Original Dataset (Before Feature Selection) ( {} ) : {:.4f}".format(ObjFunc.__name__, before_precision))
                                                        print("^^^Recall of Original Dataset (Before Feature Selection) ( {} ) : {:.4f}".format(ObjFunc.__name__, before_recall))
                                                        print("^^^F1_score of Original Dataset (Before Feature Selection) ( {} ) : {:.4f}".format(ObjFunc.__name__, before_f1_score))
                                                        # print("^^^Specificity of Original Dataset (Before Feature Selection) ( {} ) : {:.4f}".format(ObjFunc.__name__, before_specificity))
                                                        # print("^^^CCI of Original Dataset (Before Feature Selection) ( {} ) : {:}".format(ObjFunc.__name__, before_CCI))
                                                        # print("^^^ICI of Original Dataset (Before Feature Selection) ( {} ) : {:}".format(ObjFunc.__name__, before_ICI))
                                                        # print("^^^ROC of Original Dataset (Before Feature Selection) ( {} ) : {:.4f}".format(ObjFunc.__name__, before_ROC))
                                                        # print("^^^MCC of Original Dataset (Before Feature Selection) ( {} ) : {:.4f}".format(ObjFunc.__name__, before_MCC))
                                                                
                                                                
                                                        param_result_path = getcwd() + "/Final_BNRO/Final_BNRO_result.doc"

                                                        if  path.exists(param_result_path):
                                                            param_file=open(param_result_path,'a')
                                                            param_file.write("-------------------------------------------------------------------------")
                                                            param_file.write("\nOriginal Dataset (Before Feature Selection)\n\n")                                                                    
                                                            # param_file.write("\nAlgorithm:  {} >>>>\n\n".format(name))
                                                            param_file.write(">>>> Dataset:  {}\n     Objective Function:  {}  ==>   {}\n     Transfer Function:   {}  ==>  {}\n\n    Dataset Shape (Samples ==> {}, Features ==> {})\n     Training Dataset (Samples ==> {}, Features ==> {})\n     Testing Dataset (Samples ==> {}, Features ==> {})\n\n"
                                                                  .format(ds, obj_fun_cat, ObjFunc.__name__, trans_func_cat, trans_func.__name__,  shape(dataset_splits[ds][0])[0] + shape(dataset_splits[ds][1])[0] , shape(dataset_splits[ds][0])[1], shape(dataset_splits[ds][0])[0], shape(dataset_splits[ds][0])[1], shape(dataset_splits[ds][1])[0], shape(dataset_splits[ds][1])[1]))    
                                                            param_file.write("-------------------------------------------------------------------------")
                                                            param_file.write("\n*Accuracy of Original Dataset (Before Feature Selection) ( {} ) : {:.4f}".format(ObjFunc.__name__, before_accuracy))
                                                            param_file.write("\n*Error rate of Original Dataset (Before Feature Selection) ( {} ) : {:.4f}".format(ObjFunc.__name__, before_err))                                                            
                                                            param_file.write("\n*Fitness of Original Dataset (Before Feature Selection) ( {} ) : {:.4f}".format(ObjFunc.__name__, before_fit))                                                            
                                                            
                                                            param_file.write("\n***Kappa of Original Dataset (Before Feature Selection) ( {} ) : {:.4f}".format(ObjFunc.__name__, before_kappa))
                                                            param_file.write("\n***Precision of Original Dataset (Before Feature Selection) ( {} ) : {:.4f}".format(ObjFunc.__name__, before_precision))
                                                            param_file.write("\n***Recall of Original Dataset (Before Feature Selection) ( {} ) : {:.4f}".format(ObjFunc.__name__, before_recall))
                                                            param_file.write("\n***F1_score of Original Dataset (Before Feature Selection) ( {} ) : {:.4f}".format(ObjFunc.__name__, before_f1_score))
                                                            # param_file.write("\n***Specificity of Original Dataset (Before Feature Selection) ( {} ) : {:.4f}".format(ObjFunc.__name__, before_specificity))
                                                            # param_file.write("\n***CCI of Original Dataset (Before Feature Selection) ( {} ) : {:}\n\n".format(ObjFunc.__name__, before_CCI))
                                                            # param_file.write("\n***ICI of Original Dataset (Before Feature Selection) ( {} ) : {:}\n\n".format(ObjFunc.__name__, before_ICI))
                                                            # param_file.write("\n***ROC of Original Dataset (Before Feature Selection) ( {} ) : {:.4f}\n\n".format(ObjFunc.__name__, before_ROC))
                                                            # param_file.write("\n***MCC of Original Dataset (Before Feature Selection) ( {} ) : {:.4f}\n\n".format(ObjFunc.__name__, before_MCC))
                                                            param_file.write("-------------------------------------------------------------------------")
                                                            param_file.close()
                                                            
                                                        for name, Algo in self.algo_dicts.items():
                                                               
                                                                     
                                                                # param_result_path = getcwd() + "/Final_BNRO/Final_BNRO_result.doc"
                                                                     
                                                                print("> Dataset: {}, Objective Function: {} ==> {}, Transfer Function: {} ==> {}, Original Dataset Shape (Samples ==> {}, Features ==> {}), Original Training Dataset (Samples ==> {}, Features ==> {}), Original Testing Dataset (Samples ==> {}, Features ==> {}), Model Name: {}"
                                                                      .format(ds, obj_fun_cat, ObjFunc.__name__, trans_func_cat, trans_func.__name__, shape(dataset_splits[ds][0])[0] + shape(dataset_splits[ds][1])[0] , shape(dataset_splits[ds][0])[1], shape(dataset_splits[ds][0])[0], shape(dataset_splits[ds][0])[1], shape(dataset_splits[ds][1])[0], shape(dataset_splits[ds][1])[1], name))
                                                                
                                                                # print("-------------------------------------------------------------------------")
                                                                # print("^^^Accuracy of Original Dataset (Before Feature Selection) {:.4f}".format(before_accuracy))
                                                                
                                                                                                                          
                                                                if  path.exists(param_result_path):
                                                                      param_file=open(param_result_path,'a')
                                                                      param_file.write("-------------------------------------------------------------------------")
                                                                      param_file.write("\nAlgorithm:  {} >>>>\n\n".format(name))
                                                                      param_file.write(">>>> Dataset:  {}\n     Objective Function:  {}  ==>   {}\n     Transfer Function:   {}  ==>  {}\n\n     Original Dataset Shape (Samples ==> {}, Features ==> {})\n     Original Training Dataset (Samples ==> {}, Features ==> {})\n     Original Testing Dataset (Samples ==> {}, Features ==> {})\n\n"
                                                                      .format(ds, obj_fun_cat, ObjFunc.__name__, trans_func_cat, trans_func.__name__,  shape(dataset_splits[ds][0])[0] + shape(dataset_splits[ds][1])[0] , shape(dataset_splits[ds][0])[1], shape(dataset_splits[ds][0])[0], shape(dataset_splits[ds][0])[1], shape(dataset_splits[ds][1])[0], shape(dataset_splits[ds][1])[1]))    
                                                                      param_file.write("-------------------------------------------------------------------------")
                                                                      # param_file.write("\n\n***Accuracy of Original Dataset (Before Feature Selection): {:.4f}\n\n".format(before_accuracy))
                                                                      param_file.close()
                                                                     
                                                                print("Algorithm:  {} >>>>".format(name))
                                                                
                                                                if  path.exists(param_result_path):
                                                                              param_file=open(param_result_path,'a')
                                                                              param_file.write("-------------------------------------------------------------------------")                                                                                                             
                                                                              param_file.write("\nAlgorithm:  {} >>>>\n\n".format(name))
                                                                              param_file.write("-------------------------------------------------------------------------")
                                                                             
                                                                              param_file.close()
                                                                     

                                                                # res = array([zeros((5), dtype=object) for _ in range(self.num_runs)])

                                                                res = array([zeros((10), dtype=object) for _ in range(self.num_runs)])


                                                                for id_runs in range(self.num_runs):
                                                                    
                                                                        print("Run: ", id_runs)
                                                                        if  path.exists(param_result_path):
                                                                              param_file=open(param_result_path,'a')
                                                                              param_file.write("-------------------------------------------------------------------------")                                                                                                             
                                                                              param_file.write("\nRun:  {} \n".format(id_runs))
                                                                              param_file.write("-------------------------------------------------------------------------")
                                                                             
                                                                              param_file.close()
                                                                        start_time = int(round(time() * 1000))
                                                                        
                                                                        
                                                                        
                                                                        
                                                                          
                                                                        md = Algo(ml._fit__, trans_func, self.problem_size, self.domain_range, self.log,
                                                                                  self.epoch, self.pop_size, self.lsa_epoch, id_runs)
                                                                        
                                                                                                                                                
                                                                        # best_pos, best_fit, loss_train = md._train__()
                                                                        best_pos, best_fit, loss_train, best_kappa, best_precision, best_recall, best_f1_score = md._train__()
                                                                        
                                                                        time_required = int(round(time() * 1000)) - start_time
                                                                        after_accuarcy = self._get_accuracy__(best_pos, best_fit)
                                                                      
                                                                        
                                                                        res[id_runs][self.ID_FIT]  = best_fit
                                                                        res[id_runs][self.ID_ACC]  = after_accuarcy
                                                                        res[id_runs][self.ID_ERROR]  = (1 - after_accuarcy)
                                                                        res[id_runs][self.ID_FEAT] = best_pos
                                                                        res[id_runs][self.ID_TIME] = time_required
                                                                        res[id_runs][self.ID_LOSS] = loss_train                                                                        
                                                                        res[id_runs][self.ID_kappa] = best_kappa
                                                                        res[id_runs][self.ID_precision] = best_precision
                                                                        res[id_runs][self.ID_recall] = best_recall
                                                                        res[id_runs][self.ID_f1_score] = best_f1_score
                                                                        # res[id_runs][self.ID_specificity] = best_specificity
                                                                        # res[id_runs][self.ID_CCI] = best_CCI
                                                                        # res[id_runs][self.ID_ICI] = best_ICI
                                                                        # res[id_runs][self.ID_ROC] = best_ROC
                                                                        # res[id_runs][self.ID_MCC] = best_MCC

                                                                        if  path.exists(param_result_path):
                                                                              param_file=open(param_result_path,'a')
                                                                              param_file.write("-------------------------------------------------------------------------")                                                                                                             
                                                                              param_file.write("-------------------------------------------------------------------------")                                                                                                             
                                                                              param_file.write("\n^^^Best Fitness of Run {} ( Algorithm: {} ) : {:.4f}\n".format(id_runs, name, best_fit))
                                                                              param_file.write("\n^^^Best Accuracy of Run {} ( Algorithm: {} ): {:.4f}\n".format(id_runs, name, after_accuarcy))
                                                                              param_file.write("\n^^^Best Error rate of Run {} ( Algorithm: {} ): {:.4f}\n".format(id_runs, name, (1 - after_accuarcy)))
                                                                              param_file.write("\n^^^Best Kappa of Run {} ( Algorithm: {} ): {:.4f}\n".format(id_runs, name, best_kappa))
                                                                              param_file.write("\n^^^Best Precision of Run {} ( Algorithm: {} ): {:.4f}\n".format(id_runs, name, best_precision))
                                                                              param_file.write("\n^^^Best Recall of Run {} ( Algorithm: {} ): {:.4f}\n".format(id_runs, name, best_recall))
                                                                              param_file.write("\n^^^Best F1_score of Run {} ( Algorithm: {} ): {:.4f}\n".format(id_runs, name, best_f1_score))
                                                                              # param_file.write("\n^^^Best Specificity of Run {} ( Algorithm: {} ): {:.4f}\n".format(id_runs, name, best_specificity))
                                                                              # param_file.write("\n^^^Best CCI of Run {} ( Algorithm: {} ): {:}\n".format(id_runs, name, best_CCI))
                                                                              # param_file.write("\n^^^Best ICI of Run {} ( Algorithm: {} ): {:}\n".format(id_runs, name, best_ICI))
                                                                              # param_file.write("\n^^^Best ROC of Run {} ( Algorithm: {} ): {:.4f}\n".format(id_runs, name, best_ROC))
                                                                              # param_file.write("\n^^^Best MCC of Run {} ( Algorithm: {} ): {:.4f}\n".format(id_runs, name, best_MCC))
                                                                              
                                                                              param_file.write("-------------------------------------------------------------------------")
                                                                              param_file.write("-------------------------------------------------------------------------")                                                                                                             
                                                                              
                                                                              param_file.close()
                                                                              
                                                                result, best_features, worst_features = self._sort_and_get_metrics(res, self.ID_FIT, self.ID_MIN_PROB, self.ID_MAX_PROB)

                          
                                                                  
                          
                                                                fit_list.extend(result[self.ID_FIT])
                                                                acc_list.extend(result[self.ID_ACC])
                                                                err_list.extend(result[self.ID_ERROR])
                                                                feat_list.extend(result[self.ID_FEAT])
                                                                time_list.extend(result[self.ID_TIME])
                                                                loss_list.extend([result[self.ID_LOSS]])
                                                                kappa_list.extend(result[self.ID_kappa])
                                                                precision_list.extend(result[self.ID_precision])
                                                                recall_list.extend(result[self.ID_recall])
                                                                f1_score_list.extend(result[self.ID_f1_score])
                                                                # specificity_list.extend(result[self.ID_specificity])
                                                                # CCI_list.extend(result[self.ID_CCI])
                                                                # ICI_list.extend(result[self.ID_ICI])
                                                                # ROC_list.extend(result[self.ID_ROC])
                                                                # MCC_list.extend(result[self.ID_MCC])
                                                                
                                                                
                                                                print("-------------------------------------------------------------------------")                                                               
                                                               
                                                                print("^^^After Feature Selection ( Algorithm: {} ):".format(name))
                                                                print("***Best fitness: {:.4f}, Mean fitness: {:.4f}, Worst fitness: {:.4f}, Fitness SD: {:.4f}"
                                                                      .format(*result[self.ID_FIT]))
                                                                print("***Best accuracy: {:.4f}, Mean accuracy: {:.4f}, Worst accuracy: {:.4f}, Accuracy SD: {:.4f}"
                                                                      .format(*result[self.ID_ACC]))
                                                                print("***Best Error rate: {:.4f}, Mean Error rate: {:.4f}, Worst Error rate: {:.4f}, Error rate SD: {:.4f}"
                                                                      .format(*result[self.ID_ERROR]))
                                                                
                                                                print("***Best kappa: {:.4f}, Mean kappa: {:.4f}, Worst kappa: {:.4f}, kappa SD: {:.4f}"
                                                                      .format(*result[self.ID_kappa]))                   
                                                                print("***Best precision: {:.4f}, Mean precision: {:.4f}, Worst precision: {:.4f}, Precision SD: {:.4f}"
                                                                      .format(*result[self.ID_precision]))            
                                                                print("***Best recall: {:.4f}, Mean recall: {:.4f}, Worst recall: {:.4f}, Recall SD: {:.4f}"
                                                                      .format(*result[self.ID_recall]))
                                                                print("***Best f1_score: {:.4f}, Mean f1_score: {:.4f}, Worst f1_score: {:.4f}, F1_score SD: {:.4f}"
                                                                      .format(*result[self.ID_f1_score]))
                                                                # print("***Best specificity: {:.4f}, Mean specificity: {:.4f}, Worst specificity: {:.4f}, specificity SD: {:.4f}"
                                                                #       .format(*result[self.ID_specificity]))  
                                                                # print("***Best CCI: {:}, Mean CCI: {:}, Worst CCI: {:}, CCI SD: {:.4f}"
                                                                #       .format(*result[self.ID_CCI]))  
                                                                # print("***Best ICI: {:}, Mean ICI: {:}, Worst ICI: {:}, ICI SD: {:.4f}"
                                                                #       .format(*result[self.ID_ICI]))  
                                                                # print("***Best ROC: {:.4f}, Mean ROC: {:.4f}, Worst ROC: {:.4f}, ROC SD: {:.4f}"
                                                                #       .format(*result[self.ID_ROC]))  
                                                                # print("***Best MCC: {:.4f}, Mean MCC: {:.4f}, Worst MCC: {:.4f}, MCC SD: {:.4f}"
                                                                #       .format(*result[self.ID_MCC]))  
                                                                 
                                                                print("***Best feature size: {0}/{5} ==> {6}, Mean feature size: {1:.4f}, Feature selection ratio: {4:.4f}, Worst feature size: {2}/{5} ==> {7}, Feature size SD: {3:.4f}"
                                                                      .format(*result[self.ID_FEAT], self.problem_size, best_features, worst_features))                                                                
                                                                print("***Best processing time: {:.4f}, Mean processing time: {:.4f}, Worst processing time: {:.4f}, Processing time SD: {:.4f}"
                                                                      .format(*result[self.ID_TIME]))    
                                                                
                                                                print("-------------------------------------------------------------------------")
                                                                print("-------------------------------------------------------------------------")

                                                                
                                                                if  path.exists(param_result_path):
                                                                    param_file=open(param_result_path,'a')                                                                    
                                                                    param_file.write("-------------------------------------------------------------------------")                                                                                                             
                                                                    param_file.write("\n\n^^^After Feature Selection ( Algorithm: {} ):".format(name))                                                                    
                                                                    param_file.write("\n***Best fitness: {:.4f}, Mean fitness: {:.4f}\nWorst fitness: {:.4f}, Fitness STD: {:.4f}"
                                                                          .format(*result[self.ID_FIT]))                                                                    
                                                                    param_file.write("\n\n***Best accuracy: {:.4f}, Mean accuracy: {:.4f}\nWorst accuracy: {:.4f}, Accuracy STD: {:.4f}"
                                                                          .format(*result[self.ID_ACC]))
                                                                    param_file.write("\n\n***Best Error rate: {:.4f}, Mean Error rate: {:.4f}\nWorst Error rate: {:.4f}, Error rate STD: {:.4f}"
                                                                          .format(*result[self.ID_ERROR]))
                                                                    param_file.write("\n\n***Best kappa: {:.4f}, Mean kappa: {:.4f}\nWorst kappa: {:.4f}, kappa STD: {:.4f}"
                                                                          .format(*result[self.ID_kappa]))
                                                                    param_file.write("\n\n***Best precision: {:.4f}, Mean precision: {:.4f}\nWorst precision: {:.4f}, Precision STD: {:.4f}"
                                                                          .format(*result[self.ID_precision]))
                                                                    param_file.write("\n\n***Best recall: {:.4f}, Mean recall: {:.4f}\nWorst recall: {:.4f}, Recall STD: {:.4f}"
                                                                          .format(*result[self.ID_recall]))
                                                                    param_file.write("\n\n***Best f1_score: {:.4f}, Mean f1_score: {:.4f}\nWorst f1_score: {:.4f}, F1_score STD: {:.4f}"
                                                                          .format(*result[self.ID_f1_score]))
                                                                    # param_file.write("\n\n***Best specificity: {:.4f}, Mean specificity: {:.4f}\nWorst specificity: {:.4f}, specificity STD: {:.4f}"
                                                                    #       .format(*result[self.ID_specificity]))
                                                                    # param_file.write("\n\n***Best CCI: {:}, Mean CCI: {:}\nWorst CCI: {:}, CCI STD: {:.4f}"
                                                                    #       .format(*result[self.ID_CCI]))
                                                                    # param_file.write("\n\n***Best ICI: {:}, Mean ICI: {:}\nWorst ICI: {:}, ICI STD: {:.4f}"
                                                                    #       .format(*result[self.ID_ICI]))
                                                                    # param_file.write("\n\n***Best ROC: {:.4f}, Mean ROC: {:.4f}\nWorst ROC: {:.4f}, ROC STD: {:.4f}"
                                                                    #       .format(*result[self.ID_ROC]))
                                                                    # param_file.write("\n\n***Best MCC: {:.4f}, Mean MCC: {:.4f}\nWorst MCC: {:.4f}, MCC STD: {:.4f}"
                                                                    #       .format(*result[self.ID_MCC]))
                                                                    param_file.write("\n\n***Best feature size: {0}/{5} ==> {6}, Mean feature size: {1:.4f}, Feature selection ratio: {4:.4f}\nWorst feature size: {2}/{5} ==> {7}, Feature size STD: {3:.4f}"
                                                                          .format(*result[self.ID_FEAT], self.problem_size, best_features, worst_features))
                                                                    param_file.write("\n\n***Best processing time: {:.4f}, Mean processing time: {:.4f}\nWorst processing time: {:.4f}, Processing time STD: {:.4f}\n\n"
                                                                          .format(*result[self.ID_TIME]))    
                                                                    
                                                                    param_file.write("*** Fitnesses for the 100 Iterations of the Best Run  : {:}\n\n".format(result[self.ID_LOSS]))
                                                                    param_file.write("-------------------------------------------------------------------------")                                         
                                                                    param_file.write("-------------------------------------------------------------------------")                                         
                                                                    
                                                                    param_file.close()
                                                        print("===================================================================================")

                                                       
                                                        temp_fit_dict[ds] = fit_list
                                                        temp_acc_dict[ds] = acc_list
                                                        temp_err_dict[ds] = err_list
                                                        temp_feat_dict[ds] = feat_list
                                                        temp_time_dict[ds] = time_list
                                                        temp_loss_dict[ds] = loss_list
                                                        temp_kappa_dict[ds] = kappa_list
                                                        temp_precision_dict[ds] = precision_list
                                                        temp_recall_dict[ds] = recall_list
                                                        temp_f1_score_dict[ds] = f1_score_list
                                                        # temp_specificity_dict[ds] = specificity_list
                                                        # temp_CCI_dict[ds] = CCI_list
                                                        # temp_ICI_dict[ds] = ICI_list
                                                        # temp_ROC_dict[ds] = ROC_list
                                                        # temp_MCC_dict[ds] = MCC_list

                                                        if len(self.algo_dicts) == 1:
                                                                # fit_dict[ds] = deepcopy(fit_dict[ds] + fit_list)
                                                                # acc_dict[ds] = deepcopy(acc_dict[ds] + acc_list)
                                                                # feat_dict[ds] = deepcopy(feat_dict[ds] + feat_list)
                                                                # time_dict[ds] = deepcopy(time_dict[ds] + time_list)
                                                                # loss_dict[ds] = deepcopy(loss_dict[ds] + loss_list)
                                                                
                                                                fit_dict[ds] = deepcopy(fit_dict[ds] + fit_list)
                                                                acc_dict[ds] = deepcopy(acc_dict[ds] + acc_list)
                                                                err_dict[ds] = deepcopy(err_dict[ds] + err_list)
                                                                feat_dict[ds] = deepcopy(feat_dict[ds] + feat_list)
                                                                time_dict[ds] = deepcopy(time_dict[ds] + time_list)
                                                                loss_dict[ds] = deepcopy(loss_dict[ds] + loss_list)
                                                                kappa_dict[ds] = deepcopy(kappa_dict[ds] + kappa_list)
                                                                precision_dict[ds] = deepcopy(precision_dict[ds] + precision_list)
                                                                recall_dict[ds] = deepcopy(recall_dict[ds] + recall_list)
                                                                f1_score_dict[ds] = deepcopy(f1_score_dict[ds] + f1_score_list)
                                                                # specificity_dict[ds] = deepcopy(specificity_dict[ds] + specificity_list)
                                                                # CCI_dict[ds] = deepcopy(CCI_dict[ds] + CCI_list)
                                                                # ICI_dict[ds] = deepcopy(ICI_dict[ds] + ICI_list)
                                                                # ROC_dict[ds] = deepcopy(ROC_dict[ds] + ROC_list)
                                                                # MCC_dict[ds] = deepcopy(MCC_dict[ds] + MCC_list)

                                                print("===================================================================================\n"*2)

                                                if len(self.algo_dicts) > 1:
                                                        self._output_results__(dfs=[temp_fit_dict, temp_acc_dict, temp_err_dict, temp_feat_dict, temp_time_dict, temp_kappa_dict, temp_precision_dict, temp_recall_dict, temp_f1_score_dict],
                                                                                path_list=self.path_list,
                                                                                head=[self.algo_dicts.keys()],
                                                                                out_path=[ObjFunc.__name__, trans_func.__name__])
                                                        self._output_pkls_and_figs__(temp_loss_dict, self.epoch, ObjFunc.__name__, trans_func.__name__)


                                                # if len(self.algo_dicts) > 1:
                                                #         self._output_results__(dfs=[temp_fit_dict, temp_acc_dict, temp_feat_dict, temp_time_dict],
                                                #                                path_list=self.path_list,
                                                #                                head=[self.algo_dicts.keys()],
                                                #                                out_path=[ObjFunc.__name__, trans_func.__name__])
                                                #         self._output_pkls_and_figs__(temp_loss_dict, self.epoch, ObjFunc.__name__, trans_func.__name__)
                if len(self.algo_dicts) == 1:
                        self._output_results__(dfs=[fit_dict, acc_dict, err_dict, feat_dict, time_dict, kappa_dict, precision_dict, recall_dict, f1_score_dict],
                                                path_list=[getcwd() + "/history"]*9,
                                                head=[self.algo_dicts.keys(), self.obj_funs, self.trans_funcs],
                                                out_path=[obj_fun_cat, "algorithms_overall"])
                        
                        
                # if len(self.algo_dicts) == 1:
                #         self._output_results__(dfs=[fit_dict, acc_dict, feat_dict, time_dict],
                #                                path_list=[getcwd() + "/history"]*4,
                #                                head=[self.algo_dicts.keys(), self.obj_funs, self.trans_funcs],
                #                                out_path=[obj_fun_cat, "algorithms_overall"])

