# Created by "Ahmed Gad"                                                                                %
#                                                                                                       %
#       Email:      ahmed.gad_it@yahoo.com                                                              %
#       Homepage:   https://www.researchgate.net/profile/Ahmed_G_Gad                                    %
#       Github:     https://github.com/ahmedgad19891                                                    %
#-------------------------------------------------------------------------------------------------------%

from numpy import pi, sin, cos, zeros, minimum, maximum, abs, where, sign
from numpy.random import uniform, normal, randint
from copy import deepcopy
from models.root import Root


class BinaryPSO(Root):
    """
    My binary version of Particle Swarm Optimization (PSO)
    """

    def __init__(self, objective_func=None, transfer_func=None, problem_size=1000, domain_range=(0, 1), log=True,
                 epoch=100, pop_size=10, lsa_epoch=10, seed_num=42, c1=1.2, c2=1.2, w_min=0.4, w_max=0.9):
        Root.__init__(self, objective_func, transfer_func, problem_size, domain_range, log, lsa_epoch, seed_num)
        self.epoch = epoch
        self.pop_size = pop_size
        
        self.c1 = c1            # [0-2] -> [(1.2, 1.2), (0.8, 2.0), (1.6, 0.6)]  Local and global coefficient
        self.c2 = c2
        self.w_min = w_min      # [0-1] -> [0.4-0.9]      Weight of bird
        self.w_max = w_max

    def _train__(self):
        pop = [self._create_solution__() for _ in range(self.pop_size)]
        v_max = 0.5 * (self.domain_range[1] - self.domain_range[0])
        v_list = uniform(0, v_max, (self.pop_size, self.problem_size))
        pop, g_best = self._sort_pop_and_get_global_best__(pop, self.ID_FIT, self.ID_MIN_PROB)

        for epoch in range(self.epoch):
            # Update weight after each move count  (weight down)
            w = (self.epoch - epoch) / self.epoch * (self.w_max - self.w_min) + self.w_min
            for i in range(self.pop_size):
                v_new = w * v_list[i, :] + self.c1 * uniform() * (pop[i][self.ID_POS] - pop[i][self.ID_POS]) +\
                            self.c2 * uniform() * (g_best[self.ID_POS] - pop[i][self.ID_POS])
                x_new = pop[i][self.ID_POS] + v_new             # Xi(new) = Xi(old) + Vi(new) * deltaT (deltaT = 1)
                pop[i] = self._to_binary_and_update_fit__(x_new, pop[i])

            pop, g_best = self._sort_pop_and_update_global_best__(pop, self.ID_MIN_PROB, g_best)
            self.loss_train.append(g_best[self.ID_FIT])
            if self.log:
                print("> Epoch: {}, Best fit: {}".format(epoch+1, g_best[self.ID_FIT]))

        return g_best[self.ID_POS_BIN], g_best[self.ID_FIT], self.loss_train, g_best[self.ID_kappa], g_best[self.ID_precision], g_best[self.ID_recall], g_best[self.ID_f1_score]
    
