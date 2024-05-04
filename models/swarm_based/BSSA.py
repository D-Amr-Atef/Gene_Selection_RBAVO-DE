#
# ------------------------------------------------------------------------------------------------------%
# Created by "Ahmed Gad"                                                                                %
#                                                                                                       %
#       Email:      ahmed.gad_it@yahoo.com                                                              %
#       Homepage:   https://www.researchgate.net/profile/Ahmed_G_Gad                                    %
#       Github:     https://github.com/ahmedgad19891                                                    %
#-------------------------------------------------------------------------------------------------------%

from numpy import abs, exp, ones
from numpy.random import uniform, normal
from copy import deepcopy
from models.root import Root

class BinarySSA(Root):
    """
    My Binary version of Sparrow Search Algorithm (SSA): Sparrow Search Algorithm
        (A novel swarm intelligence optimization approach: sparrow search algorithm)
    Link:
        https://doi.org/10.1080/21642583.2019.1708830
    Noted:
        + In Eq. 4, Instead of using A+ and L, I used normal(). Because at the end L*A+ is simply a random number
        + Their algorithm 1 flow is missing all important components such as g_best, fitness updated,
    """

    def __init__(self, objective_func=None, transfer_func=None, problem_size=1000, domain_range=(0, 1), log=True,
                 epoch=100, pop_size=10, lsa_epoch=10, seed_num=42):
        Root.__init__(self, objective_func, transfer_func, problem_size, domain_range, log, lsa_epoch, seed_num)
        self.epoch = epoch
        self.pop_size = pop_size
        
        self.ST = 0.8   # ST in [0.5, 1.0], safety threshold
        self.PD = 0.2   # number of producers
        self.SD = 0.1   # number of sparrows who perceive the danger
    
    def _train__(self):
        pop = [self._create_solution__() for _ in range(self.pop_size)]
        pop, g_best = self._sort_pop_and_get_global_best__(pop, self.ID_FIT, self.ID_MIN_PROB)
        n1 = int(self.PD * self.pop_size)
        n2 = int(self.SD * self.pop_size)
        
        for epoch in range(self.epoch):
            r2 = uniform()      # R2 random value in [0, 1], the alarm value

            # Using equation (3) update the sparrow’s location;
            for i in range(0, n1):
                if r2 < self.ST:
                    x_new = pop[i][self.ID_POS] * exp(-(i+1) / (uniform() * self.epoch))
                else:
                    x_new = pop[i][self.ID_POS] + normal() * ones(self.problem_size)
                pop[i] = self._to_binary_and_update_fit__(x_new, pop[i])

            x_p = deepcopy(sorted(pop[:n1], key=lambda item: item[self.ID_FIT])[0][self.ID_POS])
            worst = deepcopy(sorted(pop, key=lambda item: item[self.ID_FIT])[-1])

            # Using equation (4) update the sparrow’s location;
            for i in range(n1, self.pop_size):
                if i > int(self.pop_size / 2):
                    x_new = normal() * exp((worst[self.ID_POS] - pop[i][self.ID_POS]) / (i+1)**2)
                else:
                    x_new = x_p + abs(pop[i][self.ID_POS] - x_p) * normal()
                pop[i] = self._to_binary_and_update_fit__(x_new, pop[i])

            # Using equation (5) update the sparrow’s location;
            for i in range(0, n2):
                if pop[i][self.ID_FIT] > g_best[self.ID_FIT]:
                    x_new = g_best[self.ID_POS] + normal() * abs(pop[i][self.ID_POS] - g_best[self.ID_POS])
                else:
                    x_new = pop[i][self.ID_POS] + uniform(-1, 1) * \
                        (abs(pop[i][self.ID_POS] - worst[self.ID_POS]) / (pop[i][self.ID_FIT] - worst[self.ID_FIT] + self.EPSILON))
                pop[i] = self._to_binary_and_update_fit__(x_new, pop[i])

            pop, g_best = self._sort_pop_and_update_global_best__(pop, self.ID_MIN_PROB, g_best)
            self.loss_train.append(g_best[self.ID_FIT])
            if self.log:
                print("> Epoch: {}, Best fit: {}".format(epoch + 1, g_best[self.ID_FIT]))

        return g_best[self.ID_POS_BIN], g_best[self.ID_FIT], self.loss_train, g_best[self.ID_kappa], g_best[self.ID_precision], g_best[self.ID_recall], g_best[self.ID_f1_score]



class ImprovedBinarySSA(Root):
   

    def __init__(self, objective_func=None, transfer_func=None, problem_size=1000, domain_range=(0, 1), log=True,
                 epoch=100, pop_size=10, lsa_epoch=10, seed_num=42):
        Root.__init__(self, objective_func, transfer_func, problem_size, domain_range, log, lsa_epoch, seed_num)
        self.epoch = epoch
        self.pop_size = pop_size
        
        self.ST = 0.8   # ST in [0.5, 1.0], safety threshold
        self.PD = 0.2   # number of producers
        self.SD = 0.1   # number of sparrows who perceive the danger
        self.weighting_factor = 0.8 
        self.crossover_rate = 0.9
        
    def _train__(self):
        pop = [self._create_solution__() for _ in range(self.pop_size)]
        pop, g_best = self._sort_pop_and_get_global_best__(pop=pop, id_fitness=self.ID_FIT, id_best=self.ID_MIN_PROB,apply_DE=True, apply_lsa=False,CR_Rate=self.crossover_rate,W_Factor=self.weighting_factor) 

        n1 = int(self.PD * self.pop_size)
        n2 = int(self.SD * self.pop_size)
        
        for epoch in range(self.epoch):
            r2 = uniform()      # R2 random value in [0, 1], the alarm value

            # Using equation (3) update the sparrow’s location;
            for i in range(0, n1):
                if r2 < self.ST:
                    x_new = pop[i][self.ID_POS] * exp(-(i+1) / (uniform() * self.epoch))
                else:
                    x_new = pop[i][self.ID_POS] + normal() * ones(self.problem_size)
                pop[i] = self._to_binary_and_update_fit__(x_new, pop[i])

            x_p = deepcopy(sorted(pop[:n1], key=lambda item: item[self.ID_FIT])[0][self.ID_POS])
            worst = deepcopy(sorted(pop, key=lambda item: item[self.ID_FIT])[-1])

            # Using equation (4) update the sparrow’s location;
            for i in range(n1, self.pop_size):
                if i > int(self.pop_size / 2):
                    x_new = normal() * exp((worst[self.ID_POS] - pop[i][self.ID_POS]) / (i+1)**2)
                else:
                    x_new = x_p + abs(pop[i][self.ID_POS] - x_p) * normal()
                pop[i] = self._to_binary_and_update_fit__(x_new, pop[i])

            # Using equation (5) update the sparrow’s location;
            for i in range(0, n2):
                if pop[i][self.ID_FIT] > g_best[self.ID_FIT]:
                    x_new = g_best[self.ID_POS] + normal() * abs(pop[i][self.ID_POS] - g_best[self.ID_POS])
                else:
                    x_new = pop[i][self.ID_POS] + uniform(-1, 1) * \
                        (abs(pop[i][self.ID_POS] - worst[self.ID_POS]) / (pop[i][self.ID_FIT] - worst[self.ID_FIT] + self.EPSILON))
                pop[i] = self._to_binary_and_update_fit__(x_new, pop[i])

            pop, g_best = self._sort_pop_and_update_global_best__(pop=pop, id_best=self.ID_MIN_PROB, g_best=g_best,apply_DE=True, apply_lsa=False,CR_Rate=self.crossover_rate,W_Factor=self.weighting_factor) 
            self.loss_train.append(g_best[self.ID_FIT])
            if self.log:
                print("> Epoch: {}, Best fit: {}".format(epoch + 1, g_best[self.ID_FIT]))

        return g_best[self.ID_POS_BIN], g_best[self.ID_FIT], self.loss_train, g_best[self.ID_kappa], g_best[self.ID_precision], g_best[self.ID_recall], g_best[self.ID_f1_score]
    
class LSABinarySSA(BinarySSA):
    """
    My LSA-improved Binary version of Sparrow Search Algorithm (SSA)
    """

    def __init__(self, objective_func=None, transfer_func=None, problem_size=1000, domain_range=(0, 1), log=True,
                 epoch=100, pop_size=10, lsa_epoch=10, seed_num=42):
        Root.__init__(self, objective_func, transfer_func, problem_size, domain_range, log, lsa_epoch, seed_num)
        self.epoch = epoch
        self.pop_size = pop_size
                
        self.ST = 0.8   # ST in [0.5, 1.0], safety threshold
        self.PD = 0.2   # number of producers
        self.SD = 0.1   # number of sparrows who perceive the danger
    
    def _train__(self):
        pop = [self._create_solution__() for _ in range(self.pop_size)]
        pop, g_best = self._sort_pop_and_get_global_best__(pop, self.ID_FIT, self.ID_MIN_PROB, apply_lsa=True)
        n1 = int(self.PD * self.pop_size)
        n2 = int(self.SD * self.pop_size)
        
        for epoch in range(self.epoch):
            r2 = uniform()      # R2 random value in [0, 1], the alarm value

            # Using equation (3) update the sparrow’s location;
            for i in range(0, n1):
                if r2 < self.ST:
                    x_new = pop[i][self.ID_POS] * exp(-(i+1) / (uniform() * self.epoch))
                else:
                    x_new = pop[i][self.ID_POS] + normal() * ones(self.problem_size)
                x_new = self._amend_solution_random_faster__(x_new)
                pop[i] = self._to_binary_and_update_fit__(x_new, pop[i])

            x_p = deepcopy(sorted(pop[:n1], key=lambda item: item[self.ID_FIT])[0][self.ID_POS])
            worst = deepcopy(sorted(pop, key=lambda item: item[self.ID_FIT])[-1])

            # Using equation (4) update the sparrow’s location;
            for i in range(n1, self.pop_size):
                if i > int(self.pop_size / 2):
                    x_new = normal() * exp((worst[self.ID_POS] - pop[i][self.ID_POS]) / (i+1)**2)
                else:
                    x_new = x_p + abs(pop[i][self.ID_POS] - x_p) * normal()
                x_new = self._amend_solution_random_faster__(x_new)
                pop[i] = self._to_binary_and_update_fit__(x_new, pop[i])

            # Using equation (5) update the sparrow’s location;
            for i in range(0, n2):
                if pop[i][self.ID_FIT] > g_best[self.ID_FIT]:
                    x_new = g_best[self.ID_POS] + normal() * abs(pop[i][self.ID_POS] - g_best[self.ID_POS])
                else:
                    x_new = pop[i][self.ID_POS] + uniform(-1, 1) * \
                        (abs(pop[i][self.ID_POS] - worst[self.ID_POS]) / (pop[i][self.ID_FIT] - worst[self.ID_FIT] + self.EPSILON))
                x_new = self._amend_solution_random_faster__(x_new)
                pop[i] = self._to_binary_and_update_fit__(x_new, pop[i])

            pop, g_best = self._sort_pop_and_update_global_best__(pop, self.ID_MIN_PROB, g_best, apply_lsa=True)
            self.loss_train.append(g_best[self.ID_FIT])
            if self.log:
                print("> Epoch: {}, Best fit: {}".format(epoch + 1, g_best[self.ID_FIT]))

        return g_best[self.ID_POS_BIN], g_best[self.ID_FIT], self.loss_train, g_best[self.ID_kappa], g_best[self.ID_precision], g_best[self.ID_recall], g_best[self.ID_f1_score]

