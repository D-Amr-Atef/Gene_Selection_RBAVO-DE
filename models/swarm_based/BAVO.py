import random
import numpy as np
rng = np.random.default_rng()
import math
# from numpy import linalg as LA
import numpy as np
from copy import deepcopy
from models.root import Root

class BinaryAVO(Root):
    """
    The Improved Binary version of: African-Vulture-Optimization (AVO) algorithm
   
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
      
    ##########################################################################
  
        self.nfe_per_epoch = pop_size
        self.sort_flag = False

        self.epoch = epoch
        self.pop_size = pop_size      
        
        self.problem_size = problem_size
       
        self.rng = np.random.default_rng()
      
        ##########################################################################
        self.alpha=0.8
        self.betha=0.2
        self.p1 = 0.6
        self.p2=0.4
        self.p3=0.6
        self.Gama = 2.5
        
    ##########################################################################
    ##########################################################################   
    
    def rouletteWheelSelection(self,x):
       CS  = np.cumsum(x)
       Random_value = random.random()
       index = np.where(Random_value <= CS)
       index = sum(index)
       return index

    ##########################################################################
    ##########################################################################   
    
    def random_select(self,Pbest_Vulture_1,Pbest_Vulture_2,alpha,betha):
       probabilities=[alpha, betha]
       index = self.rouletteWheelSelection(probabilities)
       if ( index.all()> 0):
            random_vulture_X=Pbest_Vulture_1
       else:
            random_vulture_X=Pbest_Vulture_2
    
       return random_vulture_X

    ##########################################################################
    ##########################################################################     
    
    # eq (18) 
    def get_simple_levy_step(self):
       beta = 3/2
       sigma = (math.gamma(1 + beta) * math.sin(math.pi * beta / 2) / (math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
       u = np.random.normal(0, 1, self.problem_size) * sigma
       v = np.random.normal(1, self.problem_size)
       step = u / abs(v) ** (1 / beta)
       return step
    ##########################################################################
    ##########################################################################   
    
    def exploration(self,current_vulture_X, random_vulture_X, F, p1, upper_bound, lower_bound):
       if random.random()<p1:
          current_vulture_X=random_vulture_X-(abs((2*random.random())*random_vulture_X-current_vulture_X))*F;
       else:
          current_vulture_X=(random_vulture_X-(F)+random.random()*((upper_bound-lower_bound)*random.random()+lower_bound));
       return current_vulture_X

    ##########################################################################
    ##########################################################################   

    def exploitation(self,current_vulture_X, Best_vulture1_X, Best_vulture2_X,random_vulture_X, F, p2, p3, variables_no, upper_bound, lower_bound):
       if  abs(F)<0.5:
          
         if random.random()<p2:
            
             A = Best_vulture1_X-((np.multiply(Best_vulture1_X,current_vulture_X))/(Best_vulture1_X-current_vulture_X**2))*F
             B = Best_vulture2_X-((Best_vulture2_X*current_vulture_X)/(Best_vulture2_X-current_vulture_X**2))*F
             current_vulture_X=(A+B)/2
         
         else:
             current_vulture_X = random_vulture_X-abs(random_vulture_X-current_vulture_X)*F*self.get_simple_levy_step()
                        
       if random.random()>=0.5:
         if random.random()<p3:
             current_vulture_X = (abs((2*random.random())*random_vulture_X-current_vulture_X))*(F+random.random())-(random_vulture_X-current_vulture_X)
         else:
            s1 = random_vulture_X*(random.random()*current_vulture_X/(2*math.pi))*np.cos(current_vulture_X)
            s2 = random_vulture_X*(random.random()*current_vulture_X/(2*math.pi))*np.sin(current_vulture_X)
            current_vulture_X=random_vulture_X-(s1+s2)
       return current_vulture_X

    ##########################################################################
    ##########################################################################   

    def AVO(self, pop, current_iter):
        Pbest_Vulture_1  = pop[0]                     #location of Vulture (First best location Best Vulture Category 1) 
        Pbest_Vulture_2  = pop[1]                     #location of Vulture (Second best location Best Vulture Category 1)
        t3 = np.random.uniform(-2,2,1)*((np.sin((math.pi/2)*(current_iter/self.epoch))**self.Gama)+np.cos((math.pi/2)*(current_iter/self.epoch))-1)
        # z = random.randint(-1, 0)
        P1 = (2*random.random()+1)*(1-(current_iter/self.epoch))+t3
        F = P1*(2*random.random()-1)
        
        pop_new = []
        
        
        for idx in range(0, self.pop_size):    
       
             current_vulture_X = deepcopy(pop[idx])[self.ID_POS]
             
             random_vulture_X = self.random_select(Pbest_Vulture_1,Pbest_Vulture_2,self.alpha,self.betha)   # select random vulture using eq(1)
             
             if abs(F) >=1:
                  current_vulture_X = self.exploration(current_vulture_X, random_vulture_X[self.ID_POS], F, self.p1, self.domain_range[1], self.domain_range[0]) # eq (16) & (17)         
             else:
                  
                  current_vulture_X = self.exploitation(current_vulture_X, Pbest_Vulture_1[self.ID_POS], Pbest_Vulture_2[self.ID_POS], random_vulture_X[self.ID_POS], F, self.p2, self.p3, self.problem_size, self.domain_range[1], self.domain_range[0]) # eq (10) & (13)             
             pos_new = current_vulture_X 
                         
             pop_new.append(self._to_binary_and_update_fit__(pos_new, pop[idx]))
             
        return pop_new  
    ##########################################################################
    ##########################################################################
    def _train__(self): 
                
    
        pop = [self._create_solution__() for _ in range(self.pop_size)]

        # pop, g_best = self._sort_pop_and_get_global_best__(pop=pop, id_fitness=self.ID_FIT, id_best=self.ID_MIN_PROB,apply_lsa=True)
      
        pop, g_best = self._sort_pop_and_get_global_best__(pop=pop, id_fitness=self.ID_FIT, id_best=self.ID_MIN_PROB,apply_DE=False, apply_lsa=False,CR_Rate=None,W_Factor=None) 

        for current_iter in range(1,self.epoch):
            # print("Iteration_AVO: ", current_iter)
            
            pop = self.AVO(pop, current_iter)  
                    
            ##########################################################################
            ##########################################################################
            
            # pop, g_best = self._sort_pop_and_update_global_best__(pop=pop, id_best=self.ID_MIN_PROB, g_best=g_best,apply_lsa=True)
            pop, g_best = self._sort_pop_and_update_global_best__(pop=pop, id_best=self.ID_MIN_PROB, g_best=g_best,apply_DE=False,apply_lsa=False,CR_Rate=None,W_Factor=None) 

            self.loss_train.append(g_best[self.ID_FIT])
            if self.log:
                print("> Epoch: {}, Best fit: {}".format(current_iter + 1, g_best[self.ID_FIT]))
  
        #------------------------------------------------------
        #------------------------------------------------------
        return g_best[self.ID_POS_BIN], g_best[self.ID_FIT], self.loss_train, g_best[self.ID_kappa], g_best[self.ID_precision], g_best[self.ID_recall], g_best[self.ID_f1_score]
        #------------------------------------------------------
        #------------------------------------------------------          


##########################################################################
##########################################################################
##########################################################################
class ImprovedBinaryAVO(Root):
    """
    The Improved Binary version of: African-Vulture-Optimization (AVO) algorithm
   
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
    
    def __init__(self, objective_func=None, transfer_func=None, problem_size=50, domain_range=(-1, 1), log=True,
                  epoch=100, pop_size=10, lsa_epoch=10, seed_num=42):

        Root.__init__(self, objective_func, transfer_func, problem_size, domain_range, log, lsa_epoch, seed_num)
    ##########################################################################
  
        self.nfe_per_epoch = pop_size
        self.sort_flag = False

        self.epoch = epoch
        self.pop_size = pop_size      
        
        self.problem_size = problem_size
       
        self.rng = np.random.default_rng()
      
        ##########################################################################
        self.alpha=0.8
        self.betha=0.2
        self.p1 = 0.6
        self.p2=0.4
        self.p3=0.6
        self.Gama = 2.5
        ###################################################################
        ###################################################################       
        self.weighting_factor = 0.8 
        self.crossover_rate = 0.9
    ##########################################################################
    ##########################################################################   
    
    def rouletteWheelSelection(self,x):
       CS  = np.cumsum(x)
       Random_value = random.random()
       index = np.where(Random_value <= CS)
       index = sum(index)
       return index

    ##########################################################################
    ##########################################################################   
    
    def random_select(self,Pbest_Vulture_1,Pbest_Vulture_2,alpha,betha):
       probabilities=[alpha, betha]
       index = self.rouletteWheelSelection(probabilities)
       if ( index.all()> 0):
            random_vulture_X=Pbest_Vulture_1
       else:
            random_vulture_X=Pbest_Vulture_2
    
       return random_vulture_X

    ##########################################################################
    ##########################################################################     
    
    # eq (18) 
    def get_simple_levy_step(self):
       beta = 3/2
       sigma = (math.gamma(1 + beta) * math.sin(math.pi * beta / 2) / (math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
       u = np.random.normal(0, 1, self.problem_size) * sigma
       v = np.random.normal(1, self.problem_size)
       step = u / abs(v) ** (1 / beta)
       return step
    ##########################################################################
    ##########################################################################   
    
    def exploration(self,current_vulture_X, random_vulture_X, F, p1, upper_bound, lower_bound):
       if random.random()<p1:
          current_vulture_X=random_vulture_X-(abs((2*random.random())*random_vulture_X-current_vulture_X))*F;
       else:
          current_vulture_X=(random_vulture_X-(F)+random.random()*((upper_bound-lower_bound)*random.random()+lower_bound));
       return current_vulture_X

    ##########################################################################
    ##########################################################################   

    def exploitation(self,current_vulture_X, Best_vulture1_X, Best_vulture2_X,random_vulture_X, F, p2, p3, variables_no, upper_bound, lower_bound):
       if  abs(F)<0.5:
          
         if random.random()<p2:
            
             A = Best_vulture1_X-((np.multiply(Best_vulture1_X,current_vulture_X))/(Best_vulture1_X-current_vulture_X**2))*F
             B = Best_vulture2_X-((Best_vulture2_X*current_vulture_X)/(Best_vulture2_X-current_vulture_X**2))*F
             current_vulture_X=(A+B)/2
         
         else:
             current_vulture_X = random_vulture_X-abs(random_vulture_X-current_vulture_X)*F*self.get_simple_levy_step()
                        
       if random.random()>=0.5:
         if random.random()<p3:
             current_vulture_X = (abs((2*random.random())*random_vulture_X-current_vulture_X))*(F+random.random())-(random_vulture_X-current_vulture_X)
         else:
            s1 = random_vulture_X*(random.random()*current_vulture_X/(2*math.pi))*np.cos(current_vulture_X)
            s2 = random_vulture_X*(random.random()*current_vulture_X/(2*math.pi))*np.sin(current_vulture_X)
            current_vulture_X=random_vulture_X-(s1+s2)
       return current_vulture_X

    ##########################################################################
    ##########################################################################   

    def AVO(self, pop, current_iter):
        Pbest_Vulture_1  = pop[0]                     #location of Vulture (First best location Best Vulture Category 1) 
        Pbest_Vulture_2  = pop[1]                     #location of Vulture (Second best location Best Vulture Category 1)
        t3 = np.random.uniform(-2,2,1)*((np.sin((math.pi/2)*(current_iter/self.epoch))**self.Gama)+np.cos((math.pi/2)*(current_iter/self.epoch))-1)
        # z = random.randint(-1, 0)
        P1 = (2*random.random()+1)*(1-(current_iter/self.epoch))+t3
        F = P1*(2*random.random()-1)
        
        pop_new = []
        
        
        for idx in range(0, self.pop_size):    
       
             current_vulture_X = deepcopy(pop[idx])[self.ID_POS]
             
             random_vulture_X = self.random_select(Pbest_Vulture_1,Pbest_Vulture_2,self.alpha,self.betha)   # select random vulture using eq(1)
             
             if abs(F) >=1:
                  current_vulture_X = self.exploration(current_vulture_X, random_vulture_X[self.ID_POS], F, self.p1, self.domain_range[1], self.domain_range[0]) # eq (16) & (17)         
             else:
                  
                  current_vulture_X = self.exploitation(current_vulture_X, Pbest_Vulture_1[self.ID_POS], Pbest_Vulture_2[self.ID_POS], random_vulture_X[self.ID_POS], F, self.p2, self.p3, self.problem_size, self.domain_range[1], self.domain_range[0]) # eq (10) & (13)             
             pos_new = current_vulture_X 
              
             pos_new = self._amend_solution_random_faster__(pos_new)
           
             pop_new.append(self._to_binary_and_update_fit__(pos_new, pop[idx]))
             
        return pop_new  
    ##########################################################################
    ##########################################################################
    def _train__(self): 
                
    
        pop = [self._create_solution__() for _ in range(self.pop_size)]        

        # pop, g_best = self._sort_pop_and_get_global_best__(pop=pop, id_fitness=self.ID_FIT, id_best=self.ID_MIN_PROB,apply_lsa=True)
      
        pop, g_best = self._sort_pop_and_get_global_best__(pop=pop, id_fitness=self.ID_FIT, id_best=self.ID_MIN_PROB,apply_DE=True, apply_lsa=True,CR_Rate=self.crossover_rate,W_Factor=self.weighting_factor) 

        for current_iter in range(1,self.epoch):
            # print("Iteration_AVO: ", current_iter)
            
            pop = self.AVO(pop, current_iter)  
                    
            ##########################################################################
            ##########################################################################
            
            # pop, g_best = self._sort_pop_and_update_global_best__(pop=pop, id_best=self.ID_MIN_PROB, g_best=g_best,apply_lsa=True)
            pop, g_best = self._sort_pop_and_update_global_best__(pop=pop, id_best=self.ID_MIN_PROB, g_best=g_best,apply_DE=True,apply_lsa=True,CR_Rate=self.crossover_rate,W_Factor=self.weighting_factor) 

            self.loss_train.append(g_best[self.ID_FIT])
            if self.log:
                print("> Epoch: {}, Best fit (Current Run): {}".format(current_iter + 1, g_best[self.ID_FIT]))
        # print("> Epoch: {}, Best fit (Current Run): {}".format(current_iter + 1, g_best[self.ID_FIT]))

        #------------------------------------------------------
        #------------------------------------------------------
        return g_best[self.ID_POS_BIN], g_best[self.ID_FIT], self.loss_train, g_best[self.ID_kappa], g_best[self.ID_precision], g_best[self.ID_recall], g_best[self.ID_f1_score]
        #------------------------------------------------------
        #------------------------------------------------------          

