#
# ------------------------------------------------------------------------------------------------------%
# Created by "Ahmed Gad"                                                                                %
#                                                                                                       %
#       Email:      ahmed.gad_it@yahoo.com                                                              %
#       Homepage:   https://www.researchgate.net/profile/Ahmed_G_Gad                                    %
#       Github:     https://github.com/ahmedgad19891                                                    %
#-------------------------------------------------------------------------------------------------------%

from numpy import abs, zeros, ones
from numpy.random import uniform, choice
from copy import deepcopy
from models.root import Root

class BinarySFO(Root):
    """
    My binary version of the Sailfish Optimizer: A novel nature-inspired metaheuristic algorithm for solving
        constrained engineering optimization problems
    """

    def __init__(self, objective_func=None, transfer_func=None, problem_size=1000, domain_range=(0, 1), log=True,
                 epoch=100, pop_size=10, lsa_epoch=10, seed_num=42, pp=0.1, A=4, epxilon=0.0001):
        Root.__init__(self, objective_func, transfer_func, problem_size, domain_range, log, lsa_epoch, seed_num)
        self.epoch = epoch
        self.pop_size = pop_size        # SailFish pop size
        
        self.pp = pp                    # the rate between SailFish and Sardines (N_sf = N_s * pp) = 0.25, 0.2, 0.1
        self.A = A                      # A = 4, 6,...
        self.epxilon = epxilon          # = 0.0001, 0.001

    def _train__(self):
        s_size = int(self.pop_size / self.pp)
        sf_pop = [self._create_solution__() for _ in range(self.pop_size)]
        s_pop = [self._create_solution__() for _ in range(s_size)]
        sf_pop, sf_gbest = self._sort_pop_and_get_global_best__(sf_pop, self.ID_FIT, self.ID_MIN_PROB)
        s_pop, s_gbest = self._sort_pop_and_get_global_best__(s_pop, self.ID_FIT, self.ID_MIN_PROB)

        for epoch in range(self.epoch):

            ## Calculate lamda_i using Eq.(7)
            ## Update the position of sailfish using Eq.(6)
            for i in range(self.pop_size):
                PD = 1 - len(sf_pop) / ( len(sf_pop) + len(s_pop) )
                lamda_i = 2 * uniform() * PD - PD
                sf_pop[i][self.ID_POS] = s_gbest[self.ID_POS] - lamda_i * ( uniform() *
                                        ( sf_gbest[self.ID_POS] + s_gbest[self.ID_POS] ) / 2 - sf_pop[i][self.ID_POS] )

            ## Calculate AttackPower using Eq.(10)
            AP = self.A * ( 1 - 2 * (epoch + 1) * self.epxilon )
            if AP < 0.5:
                alpha = int(len(s_pop) * abs(AP) )
                beta = int(self.problem_size * abs(AP))
                ### Random choice number of sardines which will be updated their position
                list1 = choice(range(len(s_pop)), alpha)
                for i in range(len(s_pop)):
                    if i in list1:
                        #### Random choice number of dimensions in sardines updated
                        list2 = choice(range(self.problem_size), beta)
                        for j in range(self.problem_size):
                            if j in list2:
                                ##### Update the position of selected sardines and selected their dimensions
                                s_pop[i][self.ID_POS][j] = uniform()*( sf_gbest[self.ID_POS][j] - s_pop[i][self.ID_POS][j] + AP )
            else:
                ### Update the position of all sardine using Eq.(9)
                for i in range(len(s_pop)):
                    s_pop[i][self.ID_POS] = uniform()*( sf_gbest[self.ID_POS] - s_pop[i][self.ID_POS] + AP )

            ## Recalculate the fitness
            for i in range(len(s_pop)):
                s_pop[i] = self._to_binary_and_update_fit__(s_pop[i][self.ID_POS], s_pop[i])
            
            ## Sort the population of sailfish and sardine (for reducing computational cost)
            sf_pop = sorted(sf_pop, key=lambda temp: (temp[self.ID_FIT], sum(temp[self.ID_POS_BIN])))
            s_pop = sorted(s_pop, key=lambda temp: (temp[self.ID_FIT], sum(temp[self.ID_POS_BIN])))
            for i in range(self.pop_size):
                for j in range(len(s_pop)):
                    ### If there is a better solution in sardine population.
                    if sf_pop[i][self.ID_FIT] > s_pop[j][self.ID_FIT]:
                        sf_pop[i] = deepcopy(s_pop[j])
                        del s_pop[j]
                    break   #### This simple keyword helped reducing ton of comparing operation.
                            #### Especially when sardine pop size >> sailfish pop size

            s_temp = [self._create_solution__() for _ in range(s_size - len(s_pop))]
            s_pop = s_pop + s_temp

            sf_pop, sf_gbest = self._sort_pop_and_update_global_best__(sf_pop, self.ID_MIN_PROB, sf_gbest)
            s_pop, s_gbest = self._sort_pop_and_update_global_best__(s_pop, self.ID_MIN_PROB, s_gbest)
            self.loss_train.append(sf_gbest[self.ID_FIT])
            if self.log:
                print("> Epoch: {}, Best fit: {}".format(epoch + 1, sf_gbest[self.ID_FIT]))

        return sf_gbest[self.ID_POS_BIN], sf_gbest[self.ID_FIT], self.loss_train, sf_gbest[self.ID_kappa], sf_gbest[self.ID_precision], sf_gbest[self.ID_recall], sf_gbest[self.ID_f1_score]



class ImprovedBinarySFO(Root):
    

    def __init__(self, objective_func=None, transfer_func=None, problem_size=1000, domain_range=(0, 1), log=True,
                 epoch=100, pop_size=10, lsa_epoch=10, seed_num=42, pp=0.1, A=4, epxilon=0.0001):
        Root.__init__(self, objective_func, transfer_func, problem_size, domain_range, log, lsa_epoch, seed_num)
        self.epoch = epoch
        self.pop_size = pop_size        # SailFish pop size
        
        self.pp = pp                    # the rate between SailFish and Sardines (N_sf = N_s * pp) = 0.25, 0.2, 0.1
        self.A = A                      # A = 4, 6,...
        self.epxilon = epxilon          # = 0.0001, 0.001

        self.weighting_factor = 0.8 
        self.crossover_rate = 0.9
        
    def _train__(self):
        s_size = int(self.pop_size / self.pp)
        sf_pop = [self._create_solution__() for _ in range(self.pop_size)]
        s_pop = [self._create_solution__() for _ in range(s_size)]
        sf_pop, sf_gbest = self._sort_pop_and_get_global_best__(pop=sf_pop, id_fitness=self.ID_FIT, id_best=self.ID_MIN_PROB,apply_DE=True, apply_lsa=False,CR_Rate=self.crossover_rate,W_Factor=self.weighting_factor) 
        s_pop, s_gbest = self._sort_pop_and_get_global_best__(pop=s_pop, id_fitness=self.ID_FIT, id_best=self.ID_MIN_PROB,apply_DE=True, apply_lsa=False,CR_Rate=self.crossover_rate,W_Factor=self.weighting_factor) 

        for epoch in range(self.epoch):

            ## Calculate lamda_i using Eq.(7)
            ## Update the position of sailfish using Eq.(6)
            for i in range(self.pop_size):
                PD = 1 - len(sf_pop) / ( len(sf_pop) + len(s_pop) )
                lamda_i = 2 * uniform() * PD - PD
                sf_pop[i][self.ID_POS] = s_gbest[self.ID_POS] - lamda_i * ( uniform() *
                                        ( sf_gbest[self.ID_POS] + s_gbest[self.ID_POS] ) / 2 - sf_pop[i][self.ID_POS] )

            ## Calculate AttackPower using Eq.(10)
            AP = self.A * ( 1 - 2 * (epoch + 1) * self.epxilon )
            if AP < 0.5:
                alpha = int(len(s_pop) * abs(AP) )
                beta = int(self.problem_size * abs(AP))
                ### Random choice number of sardines which will be updated their position
                list1 = choice(range(len(s_pop)), alpha)
                for i in range(len(s_pop)):
                    if i in list1:
                        #### Random choice number of dimensions in sardines updated
                        list2 = choice(range(self.problem_size), beta)
                        for j in range(self.problem_size):
                            if j in list2:
                                ##### Update the position of selected sardines and selected their dimensions
                                s_pop[i][self.ID_POS][j] = uniform()*( sf_gbest[self.ID_POS][j] - s_pop[i][self.ID_POS][j] + AP )
            else:
                ### Update the position of all sardine using Eq.(9)
                for i in range(len(s_pop)):
                    s_pop[i][self.ID_POS] = uniform()*( sf_gbest[self.ID_POS] - s_pop[i][self.ID_POS] + AP )

            ## Recalculate the fitness
            for i in range(len(s_pop)):
                s_pop[i] = self._to_binary_and_update_fit__(s_pop[i][self.ID_POS], s_pop[i])
            
            ## Sort the population of sailfish and sardine (for reducing computational cost)
            sf_pop = sorted(sf_pop, key=lambda temp: (temp[self.ID_FIT], sum(temp[self.ID_POS_BIN])))
            s_pop = sorted(s_pop, key=lambda temp: (temp[self.ID_FIT], sum(temp[self.ID_POS_BIN])))
            for i in range(self.pop_size):
                for j in range(len(s_pop)):
                    ### If there is a better solution in sardine population.
                    if sf_pop[i][self.ID_FIT] > s_pop[j][self.ID_FIT]:
                        sf_pop[i] = deepcopy(s_pop[j])
                        del s_pop[j]
                    break   #### This simple keyword helped reducing ton of comparing operation.
                            #### Especially when sardine pop size >> sailfish pop size

            s_temp = [self._create_solution__() for _ in range(s_size - len(s_pop))]
            s_pop = s_pop + s_temp

            sf_pop, sf_gbest = self._sort_pop_and_update_global_best__(sf_pop, self.ID_MIN_PROB, sf_gbest,apply_DE=True, apply_lsa=False,CR_Rate=self.crossover_rate,W_Factor=self.weighting_factor) 
            s_pop, s_gbest = self._sort_pop_and_update_global_best__(s_pop, self.ID_MIN_PROB, s_gbest,apply_DE=True, apply_lsa=False,CR_Rate=self.crossover_rate,W_Factor=self.weighting_factor) 

            self.loss_train.append(sf_gbest[self.ID_FIT])
            if self.log:
                print("> Epoch: {}, Best fit: {}".format(epoch + 1, sf_gbest[self.ID_FIT]))

        return sf_gbest[self.ID_POS_BIN], sf_gbest[self.ID_FIT], self.loss_train, sf_gbest[self.ID_kappa], sf_gbest[self.ID_precision], sf_gbest[self.ID_recall], sf_gbest[self.ID_f1_score]
