#
# ------------------------------------------------------------------------------------------------------%
# Created by "Ahmed Gad"                                                                                %
#                                                                                                       %
#       Email:      ahmed.gad_it@yahoo.com                                                              %
#       Homepage:   https://www.researchgate.net/profile/Ahmed_G_Gad                                    %
#       Github:     https://github.com/ahmedgad19891                                                    %
#-------------------------------------------------------------------------------------------------------%

from numpy import finfo, where, logical_and, maximum, minimum, sin, abs, pi, sign, ptp, min, flatnonzero,zeros,ones, clip,log,arange,floor,sum,square,sqrt,eye,matmul,power,diag,argsort,sort,array,matlib,triu,exp,linalg
from numpy.random import seed, uniform, normal, choice, randint,permutation,rand,standard_normal
from math import gamma
from copy import deepcopy
import time

class Root:
    """ This is root of all Algorithms """

    ID_MIN_PROB = 0     # min problem
    ID_MAX_PROB = -1    # max problem
    
    #''''''''''''''''''''''''''''''''''''''''''''
    ID_Mutat_PROB=0.01  # probablity of mutaion
    
    ID_Start_T= 30      # Strarting Temperature
    ID_Final_T= 0.01    # Final Temperature
    ID_Cool_R=0.5       # Cooling Rate  (0 < ID_Cool_R < 1)
    #''''''''''''''''''''''''''''''''''''''''''''
    
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
    

    EPSILON = finfo(float).eps

    OMEGA = 0.99    #weightage for accuracy and no. of features
    
    def __init__(self, objective_func=None, transfer_func=None, problem_size=1000, domain_range=[-1, 1], log=True, lsa_epoch=10, seed_num=42):
        """
        Parameters
        ----------
        objective_func :
        transfer_func :
        problem_size :
        domain_range :
        log :
        """
        seed(seed_num)
        
        self.objective_func = objective_func
        self.transfer_func = transfer_func
        self.problem_size = problem_size
        self.domain_range = domain_range
        self.log = log
        self.lsa_epoch = lsa_epoch

        self.loss_train = [] 

    # def _create_solution__(self):
    #     """
    #     Return the encoded solution with 2 element: position of solution and fitness of solution
    #     """
    #     pos = uniform(self.domain_range[0], self.domain_range[1], self.problem_size)
    #     pos_bin = self._to_binary__(pos, ones(self.problem_size))
    #     fit = self.OMEGA if sum(pos_bin) == 0 else self._fitness_model__(pos_bin)
    #     return [pos, pos_bin, fit]
 
  
    def _create_solution__(self):
        """
        Return the encoded solution with 2 element: position of solution and fitness of solution
        """
        # pos = zeros(self.problem_size)       
        # for hh in range(self.problem_size):
        #     # pos[hh] = uniform(LB[:,hh],UB[:,hh])            
            
        #     pos[hh] = rand()*(self.domain_range[1] - self.domain_range[0]) + self.domain_range[0]

        pos = uniform(self.domain_range[0], self.domain_range[1], self.problem_size)    
        pos_bin = self._to_binary__(pos, ones(self.problem_size))
        if sum(pos_bin) == 0:
            fit = self.OMEGA #, kappa, precision, recall, f1_score, specificity, CCI, ICI, ROC, MCC = self.OMEGA
            kappa = 0
            precision= 0
            recall= 0
            f1_score = 0
        else:
            fit, kappa, precision, recall, f1_score = self._fitness_model__(pos_bin)
        
        # fit = self.OMEGA if sum(pos_bin) == 0 else self._fitness_model__(pos_bin)
        return [pos, pos_bin, fit, kappa, precision, recall, f1_score]    
    
    # def _fitness_model__(self, solution=None, minmax=ID_MIN_PROB):
    #     """
    #     :param solution: 1-D numpy array
    #     :param minmax: 0- min problem, else- max problem
    #     :return:
    #     """
    #     err = self.objective_func(solution, minmax)
    #     #in case of multi objective  []
    #     solution = self.OMEGA * err + (1 - self.OMEGA) * (sum(solution) / self.problem_size)
    #     return solution if minmax == self.ID_MIN_PROB else 1.0 / (solution + self.EPSILON)


    def _fitness_model__(self, solution=None, minmax=ID_MIN_PROB):
        """
        :param solution: 1-D numpy array
        :param minmax: 0- min problem, else- max problem
        :return:
        """
      
        err, kappa, precision, recall, f1_score = self.objective_func(solution, minmax)
        
        #in case of multi objective  []
        solution = self.OMEGA * err + (1 - self.OMEGA) * (sum(solution) / self.problem_size)
        if minmax == self.ID_MIN_PROB:
            return solution, kappa, precision, recall, f1_score
        else:
            return 1.0 / (solution + self.EPSILON), kappa, precision, recall, f1_score
        # return solution if minmax == self.ID_MIN_PROB else 1.0 / (solution + self.EPSILON) 
        
    def _get_accuracy__(self, solution=None, fit=None):
        err = (fit - (1 - self.OMEGA) * (sum(solution) / self.problem_size)) / self.OMEGA
        accuracy = 1 - err
        return accuracy
        
    def _fitness_encoded__(self, encoded=None, id_pos=None, minmax=0):
        return self._fitness_model__(solution=encoded[id_pos], minmax=self.ID_MIN_PROB)

    #  return global best
    def _get_global_best__(self, pop=None, id_fitness=None, id_best=None, apply_lsa=False):
        sorted_pop = sorted(pop, key=lambda temp: (temp[id_fitness], sum(temp[self.ID_POS_BIN])))
        sorted_pop[id_best] = self._apply_lsa_to_global_best__(sorted_pop[id_best]) if apply_lsa else sorted_pop[id_best]
        return sorted_pop[id_best]

   
    def _sort_pop_and_get_global_best__(self, pop=None, id_fitness=None, id_best=None,apply_DE=False, apply_lsa=False,CR_Rate=None,W_Factor=None):
        
        sorted_pop = sorted(pop, key=lambda temp: (temp[id_fitness], sum(temp[self.ID_POS_BIN])))
        sorted_pop[id_best] = self._apply_lsa_to_global_best__(sorted_pop[id_best]) if apply_lsa else sorted_pop[id_best]
        
        if apply_DE == True:
            sorted_pop,sorted_pop[id_best] = self._apply_deff_evolution__(sorted_pop, sorted_pop[id_best],CR_Rate,W_Factor) 
           

        else:   
            sorted_pop = sorted_pop
            sorted_pop[id_best] = sorted_pop[id_best]
          
        return sorted_pop, sorted_pop[id_best]


    
    def _amend_solution__(self, solution=None):
        return maximum(self.domain_range[0], minimum(self.domain_range[1], solution))

    # Clipping in dimension
    def _amend_solution_faster__(self, solution=None):
        return clip(solution, self.domain_range[0], self.domain_range[1])

    def _amend_solution_random__(self, solution=None):
        for i in range(self.problem_size):
            if solution[i] < self.domain_range[0] or solution[i] > self.domain_range[1]:
                solution[i] = uniform(self.domain_range[0], self.domain_range[1])
        return solution

    def _amend_solution_random_faster__(self, solution=None):
        return where(logical_and(self.domain_range[0] <= solution, solution <= self.domain_range[1]), solution, uniform(self.domain_range[0],
                                                            self.domain_range[1]))

    def _update_global_best__(self, pop=None, id_best=None, g_best=None, apply_lsa=False):
        sorted_pop = sorted(pop, key=lambda temp: (temp[self.ID_FIT], sum(temp[self.ID_POS_BIN])))
        current_best = deepcopy(sorted_pop[id_best])
        g_best = current_best if current_best[self.ID_FIT] < g_best[self.ID_FIT] else g_best
        g_best = self._apply_lsa_to_global_best__(g_best) if apply_lsa else g_best
        return g_best

   


    def _sort_pop_and_update_global_best__(self, pop=None, id_best=None, g_best=None,apply_DE=False,apply_lsa=False,CR_Rate=None,W_Factor=None):
        sorted_pop = sorted(pop, key=lambda temp: (temp[self.ID_FIT], sum(temp[self.ID_POS_BIN])))
        current_best = deepcopy(sorted_pop[id_best])
        g_best = current_best if current_best[self.ID_FIT] < g_best[self.ID_FIT] else g_best
        g_best = self._apply_lsa_to_global_best__(g_best) if apply_lsa else g_best
        
        if apply_DE == True:
            sorted_pop,g_best = self._apply_deff_evolution__(sorted_pop,g_best,CR_Rate,W_Factor) 
        else:   
            sorted_pop = sorted_pop
            g_best = g_best

        sorted_pop[id_best] = g_best
        return sorted_pop, g_best
    
     
    
    def _to_binary__(self, solution_new, solution_old_bin):
        solution_new_bin = deepcopy(solution_new)
        # print("solution_new", solution_new)
        for d in range(self.problem_size):              #loop through dimensions (features)
            # print("solution_new[d]", solution_new[d])
            tf = self.transfer_func(solution_new[d])    #transfer function
            if tf > uniform():
                solution_new_bin[d] = 0 if "s_" in self.transfer_func.__name__ else 1 - solution_old_bin[d]
            else:
                solution_new_bin[d] = 1 if "s_" in self.transfer_func.__name__ else solution_old_bin[d]
        return solution_new_bin
    
  


    # def _to_binary_and_update_fit__(self, solution_new=None, l_best=None):
    #     solution_old_bin = l_best[self.ID_POS_BIN]
    #     fit_old = l_best[self.ID_FIT]
    #     solution_new_bin = self._to_binary__(solution_new, solution_old_bin)
    #     fit_new = self.OMEGA if sum(solution_new_bin) == 0 else self._fitness_model__(solution_new_bin)
    #     if fit_new < fit_old:
    #         l_best[self.ID_POS] = solution_new
    #         l_best[self.ID_POS_BIN] = solution_new_bin
    #         l_best[self.ID_FIT] = fit_new
    #     return l_best



    def _to_binary_and_update_fit__(self, solution_new=None, l_best=None):
        solution_old_bin = l_best[self.ID_POS_BIN]
        fit_old = l_best[self.ID_FIT]
        
        solution_new_bin = self._to_binary__(solution_new, solution_old_bin)
   
        if sum(solution_new_bin) == 0:
            fit_new = self.OMEGA #, kappa_new, precision_new, recall_new, f1_score_new, specificity_new, CCI_new, ICI_new, ROC_new, MCC_new = self.OMEGA
            kappa_new = 0
            precision_new= 0
            recall_new= 0
            f1_score_new = 0
        else:
            fit_new, kappa_new, precision_new, recall_new, f1_score_new = self._fitness_model__(solution_new_bin)
        
        # fit_new = self.OMEGA if sum(solution_new_bin) == 0 else self._fitness_model__(solution_new_bin)
        if fit_new < fit_old:
            l_best[self.ID_POS] = solution_new
            l_best[self.ID_POS_BIN] = solution_new_bin
            l_best[self.ID_FIT] = fit_new
            l_best[self.ID_kappa] = kappa_new
            l_best[self.ID_precision] = precision_new
            l_best[self.ID_recall] = recall_new
            l_best[self.ID_f1_score] = f1_score_new
            # l_best[self.ID_specificity] = specificity_new
            # l_best[self.ID_CCI] = CCI_new
            # l_best[self.ID_ICI] = ICI_new
            # l_best[self.ID_ROC] = ROC_new
            # l_best[self.ID_MCC] = MCC_new
        return l_best
###############################################################
###############################################################
###############################################################
# Differential Evolution : 

    def _mutation__(self, p0, p1, p2, p3,C_rate,W_fact):
        # Choose a cut point which differs 0 and chromosome-1 (first and last element)
        cut_point = randint(1, self.problem_size - 1)
        sample = []
        for i in range(self.problem_size):

            if i == cut_point or uniform() < C_rate :
                v = p1[i] + W_fact * ( p2[i] - p3[i] )
                sample.append(v)
            else :
                sample.append(p0[i])
        return array(sample)
    
    ###############################################################
    def _apply_deff_evolution__(self, sorted_pop=None, sg_best=None, Cross_rate=None,W_factor=None):

        for i in range(len(sorted_pop)):
            
            temp = choice(range(0, len(sorted_pop)), 3, replace=False)
            while i in temp:
                temp = choice(range(0, len(sorted_pop)), 3, replace=False)
            
            ###############################################
            #create new child and append in children array
            child = self._mutation__(sorted_pop[i][self.ID_POS], sorted_pop[temp[0]][self.ID_POS], sorted_pop[temp[1]][self.ID_POS], sorted_pop[temp[2]][self.ID_POS],Cross_rate,W_factor)
            ###############################################
         
            child = self._amend_solution_random_faster__(child)
            sorted_pop[i] = self._to_binary_and_update_fit__(child, sorted_pop[i])
        ###############################################
        sorted_pop, sg_best = self._sort_pop_and_update_global_best__(pop=sorted_pop, id_best=self.ID_MIN_PROB, g_best=sg_best,apply_DE=False,apply_lsa=True,CR_Rate=Cross_rate,W_Factor=W_factor) 

        return sorted_pop, sg_best 
    
    ###############################################################
    
    def _apply_lsa_to_global_best__(self, g_best=None):
        temp = deepcopy(g_best)
        for epoch in range(self.lsa_epoch):
            temp_pos = temp[self.ID_POS_BIN]
            idx = choice(self.problem_size, 4, replace=False)
            for i in idx:
                temp_pos[i] = 1 - temp_pos[i]
            # print("fit apply_lsa_to_global_best: ")
            # temp_fit = self.OMEGA if sum(temp_pos) == 0 else self._fitness_model__(temp_pos)
            if sum(temp_pos) == 0:
                temp_fit = self.OMEGA #, temp_kappa, temp_precision, temp_recall, temp_f1_score, temp_specificity, temp_CCI, temp_ICI, temp_ROC, temp_MCC = self.OMEGA
                temp_kappa = 0
                temp_precision= 0
                temp_recall= 0
                temp_f1_score = 0
            else:
                temp_fit, temp_kappa, temp_precision, temp_recall, temp_f1_score = self._fitness_model__(temp_pos)
                    
            if temp_fit < g_best[self.ID_FIT]:
                g_best[self.ID_POS_BIN] = deepcopy(temp_pos)
                g_best[self.ID_FIT] = deepcopy(temp_fit)
                g_best[self.ID_kappa] = deepcopy(temp_kappa)
                g_best[self.ID_precision] = deepcopy(temp_precision)
                g_best[self.ID_recall] = deepcopy(temp_recall)
                g_best[self.ID_f1_score] = deepcopy(temp_f1_score)
                # g_best[self.ID_specificity] = deepcopy(temp_specificity)                              
                # g_best[self.ID_CCI] = deepcopy(temp_CCI)
                # g_best[self.ID_ICI] = deepcopy(temp_ICI)
                # g_best[self.ID_ROC] = deepcopy(temp_ROC)
                # g_best[self.ID_MCC] = deepcopy(temp_MCC)
        return g_best



    def _train__(self):
        pass

    def _get_global_worst__(self, pop=None, id_fitness=None, id_worst=None):
        sorted_pop = sorted(pop, key=lambda temp: temp[id_fitness])
        return deepcopy(sorted_pop[id_worst])  
    ###########################################################################
    ######################### WDO Methods #####################################
    
    #---------------------------------------------------------------------------

    def _create_solution_WDO__(self):
        
        # Randomize population in the range of [-1,1]:(domain range)
        pos = uniform(self.domain_range[0], self.domain_range[1], self.problem_size)

        pos_bin = self._to_binary__(pos, ones(self.problem_size))
 
        ###################################################################
        # Fitness value        

        # fitness = self._fitness_model__(solution=solution)
        fitness =  self.OMEGA if sum(pos_bin) == 0 else self._fitness_model__(pos_bin)

        ###################################################################
        # Randomize Veloctity  in the range [-0.3,0.3]
        v = self.max_v * uniform(self.domain_range[0], self.domain_range[1], self.problem_size)
        
        return [pos,pos_bin, fitness, v]
    
#---------------------------------------------------------------------------

    def _to_binary_and_update_fit_WDO__(self, solution_new=None,vel_new=None,l_best=None):
        solution_old_bin = l_best[self.ID_POS_BIN]
        fit_old = l_best[self.ID_FIT]
        solution_new_bin = self._to_binary__(solution_new, solution_old_bin)
        fit_new = self.OMEGA if sum(solution_new_bin) == 0 else self._fitness_model__(solution_new_bin)
        if fit_new < fit_old:
            l_best[self.ID_POS] = solution_new
            l_best[self.ID_POS_BIN] = solution_new_bin
            l_best[self.ID_FIT] = fit_new
            l_best[self.ID_VEL] = vel_new
        return l_best
    
#---------------------------------------------------------------------------
# Two Phase Mutation
    def _apply_Two_Phase_Mutation__(self, g_best=None, id_mut_P=None):
        temp = deepcopy(g_best)
        temp_sol=deepcopy(temp[self.ID_POS_BIN])
        # print("before:",temp_sol,temp[self.ID_FIT])

        # for epoch in range(self.lsa_epoch):
            
        #------------------------------------------------------------
        # The First Mutation
        #########################################
        # Selected Features
        idx1 = [iee for iee in range(len(temp_sol)) if temp_sol[iee] == 1] 
           
        for i1 in idx1:
            temp_mutated1 = deepcopy(temp_sol)
            #######################################
            r1 = uniform()   # r1:  random value in [0, 1]
            if r1 < id_mut_P:
                temp_mutated1[i1] = 1 - temp_mutated1[i1]
                temp_fit1 = self.OMEGA if sum(temp_mutated1) == 0 else self._fitness_model__(temp_mutated1)
                # print("ddd1:",temp_mutated1,temp_fit1)

                if temp_fit1 < g_best[self.ID_FIT]:
                    g_best[self.ID_POS_BIN] = deepcopy(temp_mutated1)
                    g_best[self.ID_FIT] = deepcopy(temp_fit1)
  

        #------------------------------------------------------------
        # The Second Mutation
        #########################################
        # UnSelected Features

        idx2 = [iee for iee in range(len(temp_sol)) if temp_sol[iee] == 0] 
           
        for i2 in idx2:
            temp_mutated2 = deepcopy(temp_sol)
            #######################################
            r2 = uniform()    # r2:  random value in [0, 1]
            if r2 < id_mut_P:
                temp_mutated2[i2] = 1 - temp_mutated2[i2]
                temp_fit2 = self.OMEGA if sum(temp_mutated2) == 0 else self._fitness_model__(temp_mutated2)
                # print("ddd2:",temp_mutated2,temp_fit2)

                if temp_fit2 < g_best[self.ID_FIT]:
                    g_best[self.ID_POS_BIN] = deepcopy(temp_mutated2)
                    g_best[self.ID_FIT] = deepcopy(temp_fit2)            

        return g_best

#--------------------------------------------------------------------------- 
#simulated annealing
    def _apply_simulated_annealing__(self, g_best=None, Start_T=None, Final_T=None, Cool_R=None):
         
         S_current = deepcopy(g_best)
         S_best=deepcopy(S_current)
         # best_fit = self.OMEGA if sum(S_best[self.ID_POS_BIN]) == 0 else self._fitness_model__(S_best[self.ID_POS_BIN])
         #''''''''''''''''''''''''''''''''''''''''''''''''''''
         T=Start_T
         #''''''''''''''''''''''''''''''''''''''''''''''''''''
         # The starting temperature (Start_T) is the highest temperature, which is gradually cooled by the
         # cooling rate (Cool_R) until it reaches the final temperature (Final_T)
         while T >= Final_T:
             
             #''''''''''''''''''''''''''''''''''''''''''''''''''''
             # Two Phase Mutation
             # print("before:",(S_current[self.ID_FIT]))

             S_new=self._apply_Two_Phase_Mutation__(g_best=S_current, id_mut_P=self.ID_Mutat_PROB)
             # print("after:",(S_new[self.ID_FIT]))

             
             #''''''''''''''''''''''''''''''''''''''''''''''''''''
             if S_new[self.ID_FIT] < S_current[self.ID_FIT]:
                 S_current[self.ID_POS_BIN]=deepcopy(S_new[self.ID_POS_BIN])
                 S_current[self.ID_FIT]=deepcopy(S_new[self.ID_FIT])
                 
                 if S_new[self.ID_FIT] < S_best[self.ID_FIT]:
                   S_best[self.ID_POS_BIN]=deepcopy(S_new[self.ID_POS_BIN])
                   S_best[self.ID_FIT]=deepcopy(S_new[self.ID_FIT])
            
             #''''''''''''''''''''''''''''''''''''''''''''''''''''
             else:
                 delta_fit=S_current[self.ID_FIT]-S_new[self.ID_FIT]
                 
                 if(exp(-delta_fit/T)<=rand()):
                     S_current[self.ID_POS_BIN]=deepcopy(S_new[self.ID_POS_BIN])
                     S_current[self.ID_FIT]=deepcopy(S_new[self.ID_FIT])
             
             #''''''''''''''''''''''''''''''''''''''''''''''''''''   
             T=Cool_R*T
             #''''''''''''''''''''''''''''''''''''''''''''''''''''
         return S_best

#---------------------------------------------------------------------------
# Cross Over & Simulated Annealing with Two Phase Mutation
    def _apply_COSAM_to_global_best__(self,pop=None, g_best=None,ST=None,FT=None,CR=None):
       
       ############################################
                  
          
           
       new_pop=deepcopy(pop)
       temp = deepcopy(g_best)
       temp_sol=deepcopy(temp[self.ID_POS_BIN])
       offspring1=[]
       offspring2=[]
       
       ############################################      
       new_pop_bin=[]
       for h in range(len(new_pop)):
          new_pop_bin.append(list(new_pop[h][self.ID_POS_BIN]))
       ############################################
       
       # print("1:  ",(g_best[self.ID_POS_BIN]),(g_best[self.ID_FIT]))
       
       # Second half of population (worst solutions)
       for i in range(len(pop)//2,len(pop)):
            
    
              
          if self.lsa_epoch>len(temp_sol)-1:
               Random_len=len(temp_sol)-1
          else:
               Random_len=self.lsa_epoch
              
              
          Random_list = list(choice(arange(1,len(temp_sol)), Random_len, replace=False))

          for j in Random_list:  
             # Define Random Cross Point
             # rand_cross_point=randint(1,len(temp_sol))
             #########################################
             offspring1=deepcopy(temp_sol)
             offspring2=deepcopy(new_pop[i][self.ID_POS_BIN])
          
             #########################################
             offspring_temp=deepcopy(offspring1)
             offspring1[j:]=deepcopy(offspring2[j:])
             offspring2[j:]=deepcopy(offspring_temp[j:])
             
             #########################################
            
             ############################################
             if list(offspring1) not in new_pop_bin:
                 off_fit1 = self.OMEGA if sum(offspring1) == 0 else self._fitness_model__(offspring1)
                 new_pop.append([temp[self.ID_POS],offspring1,off_fit1, temp[self.ID_VEL]]) 
                 new_pop_bin.append(list(offspring1))
           
                          
             ############################################
             if list(offspring2) not in new_pop_bin:
                 off_fit2 = self.OMEGA if sum(offspring2) == 0 else self._fitness_model__(offspring2)
                 new_pop.append([new_pop[i][self.ID_POS],offspring2,off_fit2, new_pop[i][self.ID_VEL]])
                 new_pop_bin.append(list(offspring2))
             
            
           
       ##################################################################      
       g_best = self._update_global_best__(new_pop, self.ID_MIN_PROB, g_best, apply_lsa=False)
       # print("2:  ",(g_best[self.ID_POS_BIN]),(g_best[self.ID_FIT]))
        
       ##################################################################
       #simulated annealing
       g_best = self._apply_simulated_annealing__(g_best,ST,FT,CR) 
       # print("3:  ",(g_best[self.ID_POS_BIN]),(g_best[self.ID_FIT]))
        
       ##################################################################
       
       return g_best               
            
               
                   
#---------------------------------------------------------------------------
#  return sorted list and global best
    def _sort_pop_and_get_global_best_WDO__(self, pop=None, id_fitness=None, id_best=None,apply_COSAM=False,Start_Temp=None,Final_Temp=None,Cool_Rate=None):
        sorted_pop = sorted(pop, key=lambda temp: (temp[id_fitness], sum(temp[self.ID_POS_BIN])))
        sorted_pop[id_best] = self._apply_COSAM_to_global_best__(sorted_pop, sorted_pop[id_best],Start_Temp,Final_Temp,Cool_Rate) if apply_COSAM else sorted_pop[id_best]
        
        return sorted_pop, sorted_pop[id_best]


#---------------------------------------------------------------------------
#  return sorted list and global best
    def _sort_pop_and_update_global_best_WDO__(self, pop=None, id_best=None, g_best=None,apply_COSAM=False,Start_Temp=None,Final_Temp=None,Cool_Rate=None):
        sorted_pop = sorted(pop, key=lambda temp: (temp[self.ID_FIT], sum(temp[self.ID_POS_BIN])))
        current_best = deepcopy(sorted_pop[id_best])
        g_best = current_best if current_best[self.ID_FIT] < g_best[self.ID_FIT] else g_best
        g_best = self._apply_COSAM_to_global_best__(sorted_pop,g_best,Start_Temp,Final_Temp,Cool_Rate) if apply_COSAM else g_best
        
        sorted_pop[id_best] = g_best
        return sorted_pop, g_best
    
    