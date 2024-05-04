# Created by "Ahmed Gad"                                                                                %
#                                                                                                       %
#       Email:      ahmed.gad_it@yahoo.com                                                              %
#       Homepage:   https://www.researchgate.net/profile/Ahmed_G_Gad                                    %
#       Github:     https://github.com/ahmedgad19891                                                    %
#-------------------------------------------------------------------------------------------------------%

from numpy import zeros, exp, mean
from numpy.random import uniform, normal
from copy import deepcopy
from models.root import Root

class BinaryBA(Root):
    """
    This is my binary version of: Bat-Inspired Algorithm
    """
    ID_POS = 0      # current position
    ID_POS_BIN = 1  # current binary position
    ID_FIT = 2      # current fitness
    ID_VEL = 3      # current velocity

    def __init__(self, objective_func=None, transfer_func=None, problem_size=1000, domain_range=(0, 1), log=True,
                 epoch=100, pop_size=10, lsa_epoch=10, seed_num=42, A=0.8, r=0.95, pf=(0, 10)):
        Root.__init__(self, objective_func, transfer_func, problem_size, domain_range, log, lsa_epoch, seed_num)
        self.epoch = epoch
        self.pop_size = pop_size

        self.A = A  # (A_min, A_max): loudness
        self.r = r  # (r_min, r_max): pulse rate / emission rate
        self.pf = pf  # (pf_min, pf_max): pulse frequency

    def _train__(self):
        v = zeros(self.problem_size)    # velocity of this bird (same number of dimension of x)
        pop = [self._create_solution__()+[v] for _ in range(self.pop_size)]
        pop, g_best = self._sort_pop_and_get_global_best__(pop, self.ID_FIT, self.ID_MIN_PROB)

        for epoch in range(self.epoch):
            for i in range(self.pop_size):
                pf = self.pf[0] + (self.pf[1] - self.pf[0]) * uniform()  # Eq. 2
                v = pop[i][self.ID_VEL] + (pop[i][self.ID_POS] - g_best[self.ID_POS]) * pf  # Eq. 3
                x_new = pop[i][self.ID_POS] + v  # Eq. 4

                ## Local Search around g_best solution
                if uniform() > self.r:
                    x_new = g_best[self.ID_POS] + 0.001 * normal(self.problem_size)  # gauss
                
                if uniform() < self.A:
                    pop[i] = self._to_binary_and_update_fit__(x_new, pop[i]) + [v]

            pop, g_best = self._sort_pop_and_update_global_best__(pop, self.ID_MIN_PROB, g_best)
            self.loss_train.append(g_best[self.ID_FIT])
            if self.log:
                print("> Epoch: {}, Best fit: {}".format(epoch + 1, g_best[self.ID_FIT]))

        return g_best[self.ID_POS_BIN], g_best[self.ID_FIT], self.loss_train, g_best[self.ID_kappa], g_best[self.ID_precision], g_best[self.ID_recall], g_best[self.ID_f1_score]
    
    
    
class ImprovedBinaryBA(Root):
    """
    This is my binary version of: Bat-Inspired Algorithm
    """
    ID_POS = 0      # current position
    ID_POS_BIN = 1  # current binary position
    ID_FIT = 2      # current fitness
    ID_VEL = 3      # current velocity
    
    
    def __init__(self, objective_func=None, transfer_func=None, problem_size=1000, domain_range=(0, 1), log=True,
                 epoch=100, pop_size=10, lsa_epoch=10, seed_num=42, A=0.8, r=0.95, pf=(0, 10)):
        Root.__init__(self, objective_func, transfer_func, problem_size, domain_range, log, lsa_epoch, seed_num)
        self.epoch = epoch
        self.pop_size = pop_size

        self.A = A  # (A_min, A_max): loudness
        self.r = r  # (r_min, r_max): pulse rate / emission rate
        self.pf = pf  # (pf_min, pf_max): pulse frequency
        self.weighting_factor = 0.8 
        self.crossover_rate = 0.9
        
    def _train__(self):
        v = zeros(self.problem_size)    # velocity of this bird (same number of dimension of x)
        pop = [self._create_solution__()+[v] for _ in range(self.pop_size)]
        pop, g_best = self._sort_pop_and_get_global_best__(pop=pop, id_fitness=self.ID_FIT, id_best=self.ID_MIN_PROB,apply_DE=True, apply_lsa=False,CR_Rate=self.crossover_rate,W_Factor=self.weighting_factor) 

        for epoch in range(self.epoch):
            for i in range(self.pop_size):
                pf = self.pf[0] + (self.pf[1] - self.pf[0]) * uniform()  # Eq. 2
                v = pop[i][self.ID_VEL] + (pop[i][self.ID_POS] - g_best[self.ID_POS]) * pf  # Eq. 3
                x_new = pop[i][self.ID_POS] + v  # Eq. 4

                ## Local Search around g_best solution
                if uniform() > self.r:
                    x_new = g_best[self.ID_POS] + 0.001 * normal(self.problem_size)  # gauss
                
                if uniform() < self.A:
                    pop[i] = self._to_binary_and_update_fit__(x_new, pop[i]) + [v]

            pop, g_best = self._sort_pop_and_update_global_best__(pop=pop, id_best=self.ID_MIN_PROB, g_best=g_best,apply_DE=True, apply_lsa=False,CR_Rate=self.crossover_rate,W_Factor=self.weighting_factor) 

            self.loss_train.append(g_best[self.ID_FIT])
            if self.log:
                print("> Epoch: {}, Best fit: {}".format(epoch + 1, g_best[self.ID_FIT]))

        return g_best[self.ID_POS_BIN], g_best[self.ID_FIT], self.loss_train, g_best[self.ID_kappa], g_best[self.ID_precision], g_best[self.ID_recall], g_best[self.ID_f1_score]
    
