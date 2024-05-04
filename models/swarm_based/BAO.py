
import numpy as np
from math import gamma
# from mealpy.optimizer import Optimizer
from models.root import Root
from os import getcwd, path
from copy import deepcopy


class BinaryAO(Root):
    """
    The Binary version of: Aquila Optimization (AO)
   
    """
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
    #               epoch=100, pop_size=10, lsa_epoch=10, seed_num=42, RT=3, g=0.2, alp=0.4, c=0.4):

    #     Root.__init__(self, objective_func, transfer_func, problem_size, domain_range, log, lsa_epoch, seed_num)
        
        
        
    def __init__(self, objective_func=None, transfer_func=None, problem_size=1000, domain_range=(0, 1), log=True,
                 epoch=100, pop_size=10, lsa_epoch=10, seed_num=42):
        
        Root.__init__(self, objective_func, transfer_func, problem_size, domain_range, log, lsa_epoch, seed_num)
      
        self.nfe_per_epoch = pop_size
        self.sort_flag = False

        self.epoch = epoch
        self.pop_size = pop_size      
        
        self.problem_size = problem_size
        
        self.alpha = 0.1
        self.delta = 0.1
        
 
    ##########################################################################
    ##########################################################################        

    def get_simple_levy_step(self):
        beta = 1.5
        sigma = (gamma(1 + beta) * np.sin(np.pi * beta / 2) / (gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
        u = np.random.normal(0, 1, self.problem_size) * sigma
        v = np.random.normal(1, self.problem_size)
        step = u / abs(v) ** (1 / beta)
        return step

    ##########################################################################
    ##########################################################################
    def evolve(self, pop, g_best, current_iter):
        """
        Args:
            current_iter (int): The current iteration
        """
        g1 = 2 * np.random.rand() - 1  # Eq. 16
        g2 = 2 * (1 - current_iter / self.epoch)  # Eq. 17
        dim_list = np.array(list(range(1, self.problem_size + 1)))
        miu = 0.00565
        r0 = 10
        r = r0 + miu * dim_list
        w = 0.005
        phi0 = 3 * np.pi / 2
        phi = -w * dim_list + phi0
        x = r * np.sin(phi)  # Eq.(9)
        y = r * np.cos(phi)  # Eq.(10)
        QF = (current_iter + 1) ** ((2 * np.random.rand() - 1) / (1 - self.epoch) ** 2)  # Eq.(15)        Quality function

        pop_new = []
  
        for idx in range(0, self.pop_size):
          
            solution = deepcopy(pop[idx])
 
            x_mean = np.mean(np.array([item[self.ID_FIT] for item in pop]), axis=0)

            if (current_iter + 1) <= (2 / 3) * self.epoch:  # Eq. 3, 4
                if np.random.rand() < 0.5:
                    pos_new = g_best[self.ID_POS] * (1 - (current_iter + 1) / self.epoch) + \
                              np.random.rand() * (x_mean - g_best[self.ID_POS])
                
                else:
                    idx = np.random.choice(list(set(range(0, self.pop_size)) - {idx}))
                    pos_new = g_best[self.ID_POS] * self.get_simple_levy_step() + \
                              pop[idx][self.ID_POS] + np.random.rand() * (y - x)  # Eq. 5
                        
            else:
                if np.random.rand() < 0.5:
                    # pos_new = self.alpha * (self.g_best[self.ID_POS] - x_mean) - np.random.rand() * \
                    #           (np.random.rand() * (self.problem.ub - self.problem.lb) + self.problem.lb) * self.delta  # Eq. 13
                    pos_new = np.zeros(self.problem_size)
                    for j in range(self.problem_size):
                       pos_new[j] = self.alpha * (g_best[self.ID_POS][j] - x_mean) - np.random.rand() * \
                              (np.random.rand() * (self.domain_range[1] - self.domain_range[0]) + self.domain_range[0]) * self.delta  # Eq. 13
                else:
                    pos_new = QF * g_best[self.ID_POS] - (g2 * pop[idx][self.ID_POS] *
                            np.random.rand()) - g2 * self.get_simple_levy_step() + np.random.rand() * g1  # Eq. 14

            pop_new.append(self._to_binary_and_update_fit__(pos_new, solution))
                                       
            
        return pop_new    
     
    ##########################################################################
    ##########################################################################
    def _train__(self):               
                
        pop = [self._create_solution__() for _ in range(self.pop_size)]
        # pop, g_best = self._sort_pop_and_get_global_best__(pop=pop, id_fitness=self.ID_FIT, id_best=self.ID_MIN_PROB,apply_lsa=False)

      
        pop, g_best = self._sort_pop_and_get_global_best__(pop=pop, id_fitness=self.ID_FIT, id_best=self.ID_MIN_PROB,apply_DE=False, apply_lsa=False,CR_Rate=None,W_Factor=None) 
 
    
        for current_iter in range(1,self.epoch):
            print("Iteration_AO_Only: ", current_iter)
            
            ####################################################
            ####################################################
            pop = self.evolve(pop, g_best, current_iter)  
            
            # pop, g_best = self._sort_pop_and_update_global_best__(pop=pop, id_best=self.ID_MIN_PROB, g_best=g_best,apply_lsa=False)
            pop, g_best = self._sort_pop_and_update_global_best__(pop=pop, id_best=self.ID_MIN_PROB, g_best=g_best,apply_DE=False,apply_lsa=False,CR_Rate=None,W_Factor=None) 

            self.loss_train.append(g_best[self.ID_FIT])
            if self.log:
                print("> Epoch: {}, Best fit: {}".format(current_iter + 1, g_best[self.ID_FIT]))

       #####################################
          
  
        #------------------------------------------------------
        #------------------------------------------------------
        return g_best[self.ID_POS_BIN], g_best[self.ID_FIT], self.loss_train, g_best[self.ID_kappa], g_best[self.ID_precision], g_best[self.ID_recall], g_best[self.ID_f1_score]
        #------------------------------------------------------
        #------------------------------------------------------

##########################################################################
##########################################################################
##########################################################################
class ImprovedBinaryAO(Root):
    """
    The ImprovedBinary version of: Aquila Optimization (AO)
   
    """
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
          
   
    
    def __init__(self, objective_func=None, transfer_func=None, problem_size=1000, domain_range=(0, 1), log=True,
                 epoch=100, pop_size=10, lsa_epoch=10, seed_num=42):
        
        Root.__init__(self, objective_func, transfer_func, problem_size, domain_range, log, lsa_epoch, seed_num)
      
        self.nfe_per_epoch = pop_size
        self.sort_flag = False

        self.epoch = epoch
        self.pop_size = pop_size      
        
        self.problem_size = problem_size
        
        self.alpha = 0.1
        self.delta = 0.1
        
        ###################################################################
        ###################################################################
        self.weighting_factor = 0.8 
        self.crossover_rate = 0.9
        
        ###################################################################
        ###################################################################
  
    ##########################################################################
    ##########################################################################        

    def get_simple_levy_step(self):
        beta = 1.5
        sigma = (gamma(1 + beta) * np.sin(np.pi * beta / 2) / (gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
        u = np.random.normal(0, 1, self.problem_size) * sigma
        v = np.random.normal(1, self.problem_size)
        step = u / abs(v) ** (1 / beta)
        return step

    ##########################################################################
    ##########################################################################
    def evolve(self, pop, g_best, current_iter):
        """
        Args:
            current_iter (int): The current iteration
        """
        g1 = 2 * np.random.rand() - 1  # Eq. 16
        g2 = 2 * (1 - current_iter / self.epoch)  # Eq. 17
        dim_list = np.array(list(range(1, self.problem_size + 1)))
        miu = 0.00565
        r0 = 10
        r = r0 + miu * dim_list
        w = 0.005
        phi0 = 3 * np.pi / 2
        phi = -w * dim_list + phi0
        x = r * np.sin(phi)  # Eq.(9)
        y = r * np.cos(phi)  # Eq.(10)
        QF = (current_iter + 1) ** ((2 * np.random.rand() - 1) / (1 - self.epoch) ** 2)  # Eq.(15)        Quality function

        pop_new = []
  
        for idx in range(0, self.pop_size):
          
            solution = deepcopy(pop[idx])
 
            x_mean = np.mean(np.array([item[self.ID_FIT] for item in pop]), axis=0)

            if (current_iter + 1) <= (2 / 3) * self.epoch:  # Eq. 3, 4
                if np.random.rand() < 0.5:
                    pos_new = g_best[self.ID_POS] * (1 - (current_iter + 1) / self.epoch) + \
                              np.random.rand() * (x_mean - g_best[self.ID_POS])
                
                else:
                    idx = np.random.choice(list(set(range(0, self.pop_size)) - {idx}))
                    pos_new = g_best[self.ID_POS] * self.get_simple_levy_step() + \
                              pop[idx][self.ID_POS] + np.random.rand() * (y - x)  # Eq. 5
                        
            else:
                if np.random.rand() < 0.5:
                    # pos_new = self.alpha * (self.g_best[self.ID_POS] - x_mean) - np.random.rand() * \
                    #           (np.random.rand() * (self.problem.ub - self.problem.lb) + self.problem.lb) * self.delta  # Eq. 13
                    pos_new = np.zeros(self.problem_size)
                    for j in range(self.problem_size):
                       pos_new[j] = self.alpha * (g_best[self.ID_POS][j] - x_mean) - np.random.rand() * \
                              (np.random.rand() * (self.domain_range[1] - self.domain_range[0]) + self.domain_range[0]) * self.delta  # Eq. 13
                else:
                    pos_new = QF * g_best[self.ID_POS] - (g2 * pop[idx][self.ID_POS] *
                            np.random.rand()) - g2 * self.get_simple_levy_step() + np.random.rand() * g1  # Eq. 14
                
            # pos_new = self._amend_solution_random_faster__(pos_new)

            pop_new.append(self._to_binary_and_update_fit__(pos_new, solution))
                                       
            
        return pop_new    
     
    ##########################################################################
    ##########################################################################
    def _train__(self):               
                
        pop = [self._create_solution__() for _ in range(self.pop_size)]
        # pop, g_best = self._sort_pop_and_get_global_best__(pop=pop, id_fitness=self.ID_FIT, id_best=self.ID_MIN_PROB,apply_lsa=False)

      
        pop, g_best = self._sort_pop_and_get_global_best__(pop=pop, id_fitness=self.ID_FIT, id_best=self.ID_MIN_PROB,apply_DE=True, apply_lsa=False,CR_Rate=self.crossover_rate,W_Factor=self.weighting_factor) 
 
    
        for current_iter in range(1,self.epoch):
            print("Iteration_AO_Only: ", current_iter)
            
            ####################################################
            ####################################################
            pop = self.evolve(pop, g_best, current_iter)  
            
            # pop, g_best = self._sort_pop_and_update_global_best__(pop=pop, id_best=self.ID_MIN_PROB, g_best=g_best,apply_lsa=False)
            pop, g_best = self._sort_pop_and_update_global_best__(pop=pop, id_best=self.ID_MIN_PROB, g_best=g_best,apply_DE=True, apply_lsa=False,CR_Rate=self.crossover_rate,W_Factor=self.weighting_factor) 

            self.loss_train.append(g_best[self.ID_FIT])
            if self.log:
                print("> Epoch: {}, Best fit: {}".format(current_iter + 1, g_best[self.ID_FIT]))

       #####################################
          
  
        #------------------------------------------------------
        #------------------------------------------------------
        return g_best[self.ID_POS_BIN], g_best[self.ID_FIT], self.loss_train, g_best[self.ID_kappa], g_best[self.ID_precision], g_best[self.ID_recall], g_best[self.ID_f1_score]
        #------------------------------------------------------
        #------------------------------------------------------
    