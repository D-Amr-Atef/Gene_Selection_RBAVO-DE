#
# ------------------------------------------------------------------------------------------------------%
# Created by "Ahmed Gad"                                                                                %
#                                                                                                       %
#       Email:      ahmed.gad_it@yahoo.com                                                              %
#       Homepage:   https://www.researchgate.net/profile/Ahmed_G_Gad                                    %
#       Github:     https://github.com/ahmedgad19891                                                    %
#-------------------------------------------------------------------------------------------------------%

from numpy import abs, exp, cos, pi
from numpy.random import uniform
from copy import deepcopy
from models.root import Root

class BinaryWOA(Root):
    """
    My binary version of Whale Optimization Algorithm (WOA)
    - In this algorithm: Prey means the best solution
    """
    
    def __init__(self, objective_func=None, transfer_func=None, problem_size=1000, domain_range=(0, 1), log=True,
                 epoch=100, pop_size=10, lsa_epoch=10, seed_num=42):
        Root.__init__(self, objective_func, transfer_func, problem_size, domain_range, log, lsa_epoch, seed_num)
        self.epoch = epoch
        self.pop_size = pop_size

    def _train__(self):
        pop = [self._create_solution__() for _ in range(self.pop_size)]
        pop, g_best = self._sort_pop_and_get_global_best__(pop, self.ID_FIT, self.ID_MIN_PROB)

        for epoch in range(self.epoch):
            a = 2 - 2 * epoch / (self.epoch - 1)            # linearly decreased from 2 to 0

            for i in range(self.pop_size):
                r = uniform()
                A = 2 * a * r - a
                C = 2 * r
                l = uniform(-1, 1)
                p = 0.5
                b = 1
                if uniform() < p:
                    if abs(A) < 1:
                        D = abs(C * g_best[self.ID_POS] - pop[i][self.ID_POS])
                        new_position = g_best[self.ID_POS] - A * D
                    else :
                        #x_rand = pop[np.random.randint(self.pop_size)]         # select random 1 solution in pop
                        x_rand = self._create_solution__()
                        D = abs(C * x_rand[self.ID_POS] - pop[i][self.ID_POS])
                        new_position = x_rand[self.ID_POS] - A * D
                else:
                    D1 = abs(g_best[self.ID_POS] - pop[i][self.ID_POS])
                    new_position = D1 * exp(b * l) * cos(2 * pi * l) + g_best[self.ID_POS]
                pop[i] = self._to_binary_and_update_fit__(new_position, pop[i])
                
            pop, g_best = self._sort_pop_and_update_global_best__(pop, self.ID_MIN_PROB, g_best)
            self.loss_train.append(g_best[self.ID_FIT])
            if self.log:
                print("> Epoch: {}, Best fit: {}".format(epoch + 1, g_best[self.ID_FIT]))

        return g_best[self.ID_POS_BIN], g_best[self.ID_FIT], self.loss_train, g_best[self.ID_kappa], g_best[self.ID_precision], g_best[self.ID_recall], g_best[self.ID_f1_score]
