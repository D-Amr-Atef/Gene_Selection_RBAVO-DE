# import math
from os import getcwd, path
import numpy as nump
from numpy.random import random
from models.root import Root
from time import time
# from copy import deepcopy



###############################################################
###############################################################
# Binary version of Brown-bear Optimization  (BBO) Algorithm
###############################################################
class BinaryBBO(Root):

    ID_POS = 0      # current position
    ID_POS_BIN = 1  # current binary position
    ID_FIT = 2      # current fitness
    ID_kappa = 3
    ID_precision = 4
    ID_recall = 5
    ID_f1_score = 6
    # ID_specificity = 7
    # ID_CCI = 8
    # ID_ICI = 9
    # ID_ROC = 10
    # ID_MCC = 11
    
   
    
    # def __init__(self, objective_func=None, transfer_func=None, problem_size=50, domain_range=(-1, 1), log=True,
    #               epoch=100, pop_size=10, lsa_epoch=10, seed_num=42):

    #     Root.__init__(self, objective_func, transfer_func, problem_size, domain_range, log, lsa_epoch, seed_num)             
        
    def __init__(self, objective_func=None, transfer_func=None, problem_size=1000, domain_range=(0, 1), log=True,
                 epoch=100, pop_size=10, lsa_epoch=10, seed_num=42):
        
        Root.__init__(self, objective_func, transfer_func, problem_size, domain_range, log, lsa_epoch, seed_num)
         
        self.pop_size = pop_size
        self.sort_flag = False
        self.epoch = epoch
        self.problem_size = problem_size
        
        ###################################################################
        ###################################################################
        
    def BBO_algo(self, pop, current_iter):
        
        # For limiting out of bound solutions and setting best solution and objective function value
        for i in range(len(pop)):    
            solution = pop[i][self.ID_POS]
            Indx_max = solution > self.domain_range[1]
            Indx_min = solution < self.domain_range[0]
            solution = (solution * ~(Indx_max + Indx_min)) + self.domain_range[1] * Indx_max + self.domain_range[0] * Indx_min
            pop[i] = self._to_binary_and_update_fit__(solution, pop[i])
        
        ###################################################################    
        # Pedal Marking Behaviour
        sorted_pop = sorted(pop, key=lambda temp: (temp[self.ID_FIT], sum(temp[self.ID_POS_BIN])))

        ##################################
        P = (current_iter/self.epoch) 
        
        for ii in range(len(pop)): 
          
          newX = nump.zeros(self.problem_size)  
          ##################################  
          # Gait while walking  
          if P >= 0 and P <= 1/3:  
             # for hh in range(self.problem_size):
             newX = pop[ii][self.ID_POS] + (-P * nump.random.rand() * pop[ii][self.ID_POS])
             
          ##################################
          # Careful Stepping   
          elif P > 1/3 and P <= 2/3:  
             # for hh in range(self.problem_size): 
             Q = P * nump.random.rand()
             Step = round(1 + nump.random.rand())
             newX = pop[ii][self.ID_POS] + (Q * (sorted_pop[self.ID_MIN_PROB][self.ID_POS] - (Step * sorted_pop[self.ID_MAX_PROB][self.ID_POS])))
             
          ##################################
          # angular velocity   
          elif P > 2/3 and P <= 1: 
             # for hh in range(self.problem_size):  
             W = 2 * P * nump.pi * nump.random.rand()
             newX = pop[ii][self.ID_POS] + ((W * sorted_pop[self.ID_MIN_PROB][self.ID_POS] - nump.abs(pop[ii][self.ID_POS])) - (W * sorted_pop[self.ID_MAX_PROB][self.ID_POS] - nump.abs(pop[ii][self.ID_POS])))       
          ##################################
          pop[ii] = self._to_binary_and_update_fit__(newX, pop[ii])
          
        ###################################################################
        # Sniffing of pedal marks
        for ii in range(len(pop)): 
            
           newX = nump.zeros(self.problem_size)   
           k = nump.round(nump.random.rand() * (len(pop)-1))
           # print("k", k)
           while k == ii or k <= 0:
               k = nump.round(nump.random.rand() * (len(pop)-1))
           # print(k) 
           if pop[ii][self.ID_FIT] < pop[int(k)][self.ID_FIT]:
               r = nump.random.rand() #* nump.ones((1, 1))
               newX = pop[ii][self.ID_POS] + r * (pop[ii][self.ID_POS] - pop[int(k)][self.ID_POS])
           else:
               r = nump.random.rand() #* nump.ones((1, 1))
               newX = pop[ii][self.ID_POS] + r * (-pop[ii][self.ID_POS] + pop[int(k)][self.ID_POS])
        
           ##################################
           pop[ii] = self._to_binary_and_update_fit__(newX, pop[ii])
         
        return pop 
         
        
    ###############################################################               
    ###############################################################    
    def _train__(self):
       
            
        pop = [self._create_solution__() for _ in range(self.pop_size)]
        
        # print(pop)

        pop, g_best = self._sort_pop_and_get_global_best__(pop=pop, id_fitness=self.ID_FIT, id_best=self.ID_MIN_PROB,apply_DE=False, apply_lsa=False,CR_Rate=None,W_Factor=None) 

        
        ####################################################      
        
        for current_iter in range(self.epoch):
            
            ####################################################
            ####################################################
            
            pop = self.BBO_algo(pop, current_iter)  
            pop, g_best = self._sort_pop_and_update_global_best__(pop=pop, id_best=self.ID_MIN_PROB, g_best=g_best,apply_DE=False,apply_lsa=False,CR_Rate=None,W_Factor=None) 
        
                
            self.loss_train.append(g_best[self.ID_FIT])
            if self.log:
                print("> Epoch: {}, Best fit: {}".format(current_iter + 1, g_best[self.ID_FIT]))
  
        #------------------------------------------------------
        #------------------------------------------------------
        return g_best[self.ID_POS_BIN], g_best[self.ID_FIT], self.loss_train, g_best[self.ID_kappa], g_best[self.ID_precision], g_best[self.ID_recall], g_best[self.ID_f1_score]
        #------------------------------------------------------
        #------------------------------------------------------   
        
###############################################################
###############################################################
# Improved Binary version of Brown-bear Optimization  (BBO) Algorithm
###############################################################
class ImprovedBinaryBBO(Root):

    ID_POS = 0      # current position
    ID_POS_BIN = 1  # current binary position
    ID_FIT = 2      # current fitness
    ID_kappa = 3
    ID_precision = 4
    ID_recall = 5
    ID_f1_score = 6
    # ID_specificity = 7
    # ID_CCI = 8
    # ID_ICI = 9
    # ID_ROC = 10
    # ID_MCC = 11
    
   
    
    # def __init__(self, objective_func=None, transfer_func=None, problem_size=50, domain_range=(-1, 1), log=True,
    #               epoch=100, pop_size=10, lsa_epoch=10, seed_num=42):

    #     Root.__init__(self, objective_func, transfer_func, problem_size, domain_range, log, lsa_epoch, seed_num)             
        
    def __init__(self, objective_func=None, transfer_func=None, problem_size=1000, domain_range=(0, 1), log=True,
                 epoch=100, pop_size=10, lsa_epoch=10, seed_num=42):
        
        Root.__init__(self, objective_func, transfer_func, problem_size, domain_range, log, lsa_epoch, seed_num)
         
        self.pop_size = pop_size
        self.sort_flag = False
        self.epoch = epoch
        self.pop_size = pop_size      
        self.problem_size = problem_size
        self.weighting_factor = 0.8 
        self.crossover_rate = 0.9
        ###################################################################
        ###################################################################
        
    def BBO_algo(self, pop, current_iter):
        
        # For limiting out of bound solutions and setting best solution and objective function value
        for i in range(len(pop)):    
            solution = pop[i][self.ID_POS]
            Indx_max = solution > self.domain_range[1]
            Indx_min = solution < self.domain_range[0]
            solution = (solution * ~(Indx_max + Indx_min)) + self.domain_range[1] * Indx_max + self.domain_range[0] * Indx_min
        
            pop[i] = self._to_binary_and_update_fit__(solution, pop[i])
        
        ###################################################################    
        # Pedal Marking Behaviour
        sorted_pop = sorted(pop, key=lambda temp: (temp[self.ID_FIT], sum(temp[self.ID_POS_BIN])))

        ##################################
        P = (current_iter/self.epoch) 
        
        for ii in range(len(pop)): 
          
          newX = nump.zeros(self.problem_size)  
          ##################################  
          # Gait while walking  
          if P >= 0 and P <= 1/3:  
             # for hh in range(self.problem_size):
             newX = pop[ii][self.ID_POS] + (-P * nump.random.rand() * pop[ii][self.ID_POS])
             
          ##################################
          # Careful Stepping   
          elif P > 1/3 and P <= 2/3:  
             # for hh in range(self.problem_size): 
             Q = P * nump.random.rand()
             Step = round(1 + nump.random.rand())
             newX = pop[ii][self.ID_POS] + (Q * (sorted_pop[self.ID_MIN_PROB][self.ID_POS] - (Step * sorted_pop[self.ID_MAX_PROB][self.ID_POS])))
             
          ##################################
          # angular velocity   
          elif P > 2/3 and P <= 1: 
             # for hh in range(self.problem_size):  
             W = 2 * P * nump.pi * nump.random.rand()
             newX = pop[ii][self.ID_POS] + ((W * sorted_pop[self.ID_MIN_PROB][self.ID_POS] - nump.abs(pop[ii][self.ID_POS])) - (W * sorted_pop[self.ID_MAX_PROB][self.ID_POS] - nump.abs(pop[ii][self.ID_POS])))       
          ##################################
          pop[ii] = self._to_binary_and_update_fit__(newX, pop[ii])
          
        ###################################################################
        # Sniffing of pedal marks
        for ii in range(len(pop)): 
            
           newX = nump.zeros(self.problem_size)   
           k = nump.round(nump.random.rand() * (len(pop)-1))
           # print("k", k)
           while k == ii or k <= 0:
               k = nump.round(nump.random.rand() * (len(pop)-1))
           # print(k) 
           if pop[ii][self.ID_FIT] < pop[int(k)][self.ID_FIT]:
               r = nump.random.rand() #* nump.ones((1, 1))
               newX = pop[ii][self.ID_POS] + r * (pop[ii][self.ID_POS] - pop[int(k)][self.ID_POS])
           else:
               r = nump.random.rand() #* nump.ones((1, 1))
               newX = pop[ii][self.ID_POS] + r * (-pop[ii][self.ID_POS] + pop[int(k)][self.ID_POS])
        
           ##################################
           pop[ii] = self._to_binary_and_update_fit__(newX, pop[ii])
         
        return pop 
         
        
    ###############################################################               
    ###############################################################    
    def _train__(self):
       
            
        pop = [self._create_solution__() for _ in range(self.pop_size)]
        
        # print(pop)

        pop, g_best = self._sort_pop_and_get_global_best__(pop=pop, id_fitness=self.ID_FIT, id_best=self.ID_MIN_PROB,apply_DE=True, apply_lsa=False,CR_Rate=self.crossover_rate,W_Factor=self.weighting_factor) 

        
        ####################################################      
        
        for current_iter in range(self.epoch):
            
            ####################################################
            ####################################################
            
            pop = self.BBO_algo(pop, current_iter)  
            pop, g_best = self._sort_pop_and_update_global_best__(pop=pop, id_best=self.ID_MIN_PROB, g_best=g_best,apply_DE=True, apply_lsa=False,CR_Rate=self.crossover_rate,W_Factor=self.weighting_factor) 

                
            self.loss_train.append(g_best[self.ID_FIT])
            if self.log:
                print("> Epoch: {}, Best fit: {}".format(current_iter + 1, g_best[self.ID_FIT]))
  
        #------------------------------------------------------
        #------------------------------------------------------
        return g_best[self.ID_POS_BIN], g_best[self.ID_FIT], self.loss_train, g_best[self.ID_kappa], g_best[self.ID_precision], g_best[self.ID_recall], g_best[self.ID_f1_score]
        #------------------------------------------------------
        #------------------------------------------------------   
        
