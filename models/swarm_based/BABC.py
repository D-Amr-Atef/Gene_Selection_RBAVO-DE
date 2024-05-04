#
# ------------------------------------------------------------------------------------------------------%
# Created by "Ahmed Gad"                                                                                %
#                                                                                                       %
#       Email:      ahmed.gad_it@yahoo.com                                                              %
#       Homepage:   https://www.researchgate.net/profile/Ahmed_G_Gad                                    %
#       Github:     https://github.com/ahmedgad19891                                                    %
#-------------------------------------------------------------------------------------------------------%

from numpy import maximum, minimum
from numpy.random import uniform, randint
from copy import deepcopy
from models.root import Root

class BinaryABC(Root):
    """
        My binary version of Artificial Bee Colony (ABC)
        Originally taken from book: Clever Algorithms
        - Improved: _create_neigh_bee__
    """

    def __init__(self, objective_func=None, transfer_func=None, problem_size=1000, domain_range=(-1, 1), log=True,
                 epoch=100, pop_size=10, lsa_epoch=10, seed_num=42, couple_bees=(16, 4), patch_variables=(5.0, 0.985), sites=(3, 1)):
        Root.__init__(self, objective_func, transfer_func, problem_size, domain_range, log, lsa_epoch, seed_num)
        self.epoch = epoch
        self.pop_size = pop_size
        
        self.e_bees = couple_bees[0]                # number of bees provided for good location and other location
        self.o_bees = couple_bees[1]
        self.patch_size = patch_variables[0]        # patch_variables = patch_variables * patch_factor (0.985)
        self.patch_factor = patch_variables[1]
        self.num_sites = sites[0]                   # 3 bees (employed bees, onlookers and scouts), 1 good partition
        self.elite_sites = sites[1]

    def _create_neigh_bee__(self, individual=None, patch_size=None):
        t1 = randint(0, len(individual[self.ID_POS]) - 1)
        new_bee = deepcopy(individual[self.ID_POS])
        new_bee[t1] = (individual[self.ID_POS][t1] + uniform() * patch_size) if uniform() < 0.5 else (individual[self.ID_POS][t1] - uniform() * patch_size)        
        individual = self._to_binary_and_update_fit__(new_bee, individual)
        return individual

    def _search_neigh__(self, parent=None, neigh_size=None):
        """
        Search 1 best solution in neigh_size solution
        """
        neigh = [self._create_neigh_bee__(parent, self.patch_size) for _ in range(0, neigh_size)]
        pop, g_best = self._sort_pop_and_get_global_best__(neigh, self.ID_FIT, self.ID_MIN_PROB)
        return g_best

    def _create_scout_bees__(self, num_scouts=None):
        return [self._create_solution__() for _ in range(num_scouts)]


    def _train__(self):
        pop = [self._create_solution__() for _ in range(self.pop_size)]
        pop, g_best = self._sort_pop_and_get_global_best__(pop, self.ID_FIT, self.ID_MIN_PROB)

        for epoch in range(0, self.epoch):
            next_gen = []
            for i in range(0, self.num_sites):
                if i < self.elite_sites:
                    neigh_size = self.e_bees
                else:
                    neigh_size = self.o_bees
                next_gen.append(self._search_neigh__(pop[i], neigh_size))

            scouts = self._create_scout_bees__(self.pop_size - self.num_sites)
            pop = next_gen + scouts
            
            pop, g_best = self._sort_pop_and_update_global_best__(pop, self.ID_MIN_PROB, g_best)
            self.patch_size = self.patch_size * self.patch_factor
            self.loss_train.append(g_best[self.ID_FIT])
            if self.log:
                print("> Epoch: {}, patch_size: {}, Best fit: {}".format(epoch + 1, self.patch_size, g_best[self.ID_FIT]))

        return g_best[self.ID_POS_BIN], g_best[self.ID_FIT], self.loss_train, g_best[self.ID_kappa], g_best[self.ID_precision], g_best[self.ID_recall], g_best[self.ID_f1_score]
