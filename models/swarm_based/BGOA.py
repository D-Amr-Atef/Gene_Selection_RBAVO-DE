#
# ------------------------------------------------------------------------------------------------------%
# Created by "Ahmed Gad"                                                                                %
#                                                                                                       %
#       Email:      ahmed.gad_it@yahoo.com                                                              %
#       Homepage:   https://www.researchgate.net/profile/Ahmed_G_Gad                                    %
#       Github:     https://github.com/ahmedgad19891                                                    %
#-------------------------------------------------------------------------------------------------------%

from numpy import exp, zeros, ones, remainder, sqrt
from copy import deepcopy
from models.root import Root

class BinaryGOA(Root):
    """
    My binary version of: Grasshopper Optimization Algorithm (GOA)
        (Grasshopper Optimisation Algorithm: Theory and Application Advances in Engineering Software)
    Link:
        http://dx.doi.org/10.1016/j.advengsoft.2017.01.004
        https://www.mathworks.com/matlabcentral/fileexchange/61421-grasshopper-optimisation-algorithm-goa
    """
    
    def __init__(self, objective_func=None, transfer_func=None, problem_size=1000, domain_range=(0, 1), log=True,
                 epoch=100, pop_size=10, lsa_epoch=10, seed_num=42, c_minmax=(0.00004, 1)):
        Root.__init__(self, objective_func, transfer_func, problem_size, domain_range, log, lsa_epoch, seed_num)
        self.epoch = epoch
        self.pop_size = pop_size
        
        self.c_minmax = c_minmax

    def _s_function__(self, r_vector):
        f = 0.5
        l = 1.5
        return f * exp(-r_vector / l) - exp(-r_vector) # Eq.(2.3) in the paper

    def _distance__(self, sol_a=None, sol_b=None):
        temp = zeros(self.problem_size)
        for it in range(0, self.problem_size):
            idx = remainder(it+1, self.problem_size)
            dist = sqrt((sol_a[it] - sol_b[it]) ** 2 + (sol_a[idx] - sol_b[idx]) ** 2)
            temp[it] = dist
        return temp

    def _train__(self):
        pop = [self._create_solution__() for _ in range(self.pop_size)]
        pop, g_best = self._sort_pop_and_get_global_best__(pop, self.ID_FIT, self.ID_MIN_PROB)

        for epoch in range(self.epoch):
            # Eq.(2.8) in the paper
            c = self.c_minmax[1] - epoch * ((self.c_minmax[1] - self.c_minmax[0]) / self.epoch)
            for i in range(0, self.pop_size):
                S_i_total = zeros(self.problem_size)
                for j in range(0, self.pop_size):
                    dist = self._distance__(pop[i][self.ID_POS], pop[j][self.ID_POS])
                    r_ij_vector = (pop[i][self.ID_POS] - pop[j][self.ID_POS]) / (dist + self.EPSILON)    # xj - xi / dij in Eq.(2.7)
                    xj_xi = 2 + remainder(dist, 2)  # |xjd - xid| in Eq. (2.7)
                    ## The first part inside the big bracket in Eq. (2.7)
                    ran = (c / 2 ) * (self.domain_range[1] - self.domain_range[0]) * ones(self.problem_size)
                    s_ij = ran * self._s_function__(xj_xi) * r_ij_vector
                    S_i_total += s_ij
                    
                x_new = c * S_i_total + g_best[self.ID_POS]     # Eq. (2.7) in the paper
                pop[i] = self._to_binary_and_update_fit__(x_new, pop[i])

            pop, g_best = self._sort_pop_and_update_global_best__(pop, self.ID_MIN_PROB, g_best)
            self.loss_train.append(g_best[self.ID_FIT])
            if self.log:
                print("> Epoch: {}, Best fit: {}".format(epoch + 1, g_best[self.ID_FIT]))

        return g_best[self.ID_POS_BIN], g_best[self.ID_FIT], self.loss_train, g_best[self.ID_kappa], g_best[self.ID_precision], g_best[self.ID_recall], g_best[self.ID_f1_score]
