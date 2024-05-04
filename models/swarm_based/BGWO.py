#
# ------------------------------------------------------------------------------------------------------%
# Created by "Ahmed Gad"                                                                                %
#                                                                                                       %
#       Email:      ahmed.gad_it@yahoo.com                                                              %
#       Homepage:   https://www.researchgate.net/profile/Ahmed_G_Gad                                    %
#       Github:     https://github.com/ahmedgad19891                                                    %
#-------------------------------------------------------------------------------------------------------%

from numpy import abs
from numpy.random import uniform, seed
from copy import deepcopy
from models.root import Root

class BinaryGWO(Root):
    """
    My binary version of Grey Wolf Optimizer (GWO)
        - In this algorithms: Prey means the best solution
        https://www.mathworks.com/matlabcentral/fileexchange/44974-grey-wolf-optimizer-gwo
    """

    def __init__(self, objective_func=None, transfer_func=None, problem_size=1000, domain_range=(0, 1), log=True,
                 epoch=100, pop_size=10, lsa_epoch=10, seed_num=42):
        Root.__init__(self, objective_func, transfer_func, problem_size, domain_range, log, lsa_epoch, seed_num)
        self.epoch = epoch
        self.pop_size = pop_size

    def _train__(self):
        pop = [self._create_solution__() for _ in range(self.pop_size)]
        pop = sorted(pop, key=lambda temp: (temp[self.ID_FIT], sum(temp[self.ID_POS_BIN])))
        best_1, best_2, best_3 = deepcopy(pop[:3])

        for epoch in range(self.epoch):
            a = 2 - 2 * epoch / (self.epoch - 1)            # linearly decreased from 2 to 0

            for i in range(self.pop_size):

                A1, A2, A3 = a * (2 * uniform() - 1), a * (2 * uniform() - 1), a * (2 * uniform() - 1)
                C1, C2, C3 = 2 * uniform(), 2 * uniform(), 2 * uniform()

                X1 = best_1[self.ID_POS] - A1 * abs(C1 * best_1[self.ID_POS] - pop[i][self.ID_POS])
                X2 = best_2[self.ID_POS] - A2 * abs(C2 * best_2[self.ID_POS] - pop[i][self.ID_POS])
                X3 = best_3[self.ID_POS] - A3 * abs(C3 * best_3[self.ID_POS] - pop[i][self.ID_POS])
                temp = (X1 + X2 + X3) / 3.0
                pop[i] = self._to_binary_and_update_fit__(temp, pop[i])

            pop = sorted(pop, key=lambda temp: (temp[self.ID_FIT], sum(temp[self.ID_POS_BIN])))
            cur_best_1, cur_best_2, cur_best_3 = deepcopy(pop[:3])
            if cur_best_1[self.ID_FIT] < best_1[self.ID_FIT]:
                best_1 = deepcopy(cur_best_1)
            if cur_best_2[self.ID_FIT] < best_2[self.ID_FIT]:
                best_2 = deepcopy(cur_best_2)
            if cur_best_3[self.ID_FIT] < best_3[self.ID_FIT]:
                best_3 = deepcopy(cur_best_3)

            self.loss_train.append(best_1[self.ID_FIT])
            if self.log:
                print("> Epoch: {}, Best fit: {}".format(epoch + 1, best_1[self.ID_FIT]))

        # return best_1[self.ID_POS_BIN], best_1[self.ID_FIT], self.loss_train
        return best_1[self.ID_POS_BIN], best_1[self.ID_FIT], self.loss_train, best_1[self.ID_kappa], best_1[self.ID_precision], best_1[self.ID_recall], best_1[self.ID_f1_score]
