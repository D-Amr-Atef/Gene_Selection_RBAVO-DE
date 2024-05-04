#
# ------------------------------------------------------------------------------------------------------%
# Created by "Ahmed Gad"                                                                                %
#                                                                                                       %
#       Email:      ahmed.gad_it@yahoo.com                                                              %
#       Homepage:   https://www.researchgate.net/profile/Ahmed_G_Gad                                    %
#       Github:     https://github.com/ahmedgad19891                                                    %
#-------------------------------------------------------------------------------------------------------%

from numpy import argsort, exp, zeros
from numpy.random import uniform
from copy import deepcopy
from models.root import Root


class BinaryHGSO(Root):
    """
    My binary version of Henry gas solubility optimization: A novel physics-based algorithm
    """

    def __init__(self, objective_func=None, transfer_func=None, problem_size=1000, domain_range=(0, 1), log=True,
                 epoch=100, pop_size=10, lsa_epoch=10, seed_num=42, n_clusters = 2):
        Root.__init__(self, objective_func, transfer_func, problem_size, domain_range, log, lsa_epoch, seed_num)
        self.epoch = epoch
        self.pop_size = pop_size
        
        self.n_clusters = n_clusters
        self.n_elements = int(self.pop_size / self.n_clusters)

        self.T0 = 298.15
        self.K = 1.0
        self.beta = 1.0
        self.alpha = 1
        self.epxilon = 0.05

        self.l1 = 5E-3
        self.l2 = 1E+2
        self.l3 = 1E-2
        self.H_j = self.l1 * uniform()
        self.P_ij = self.l2 * uniform()
        self.C_j = self.l3 * uniform()

    def _create_population__(self, n_clusters=0):
        pop = []
        group = []
        for i in range(n_clusters):
            team = []
            for j in range(self.n_elements):
                temp = self._create_solution__()
                team.append(temp)
                pop.append(temp)
            group.append(team)
        return pop, group

    def _sort_group_and_get_best_solution_in_team(self, group=None):
        sorted_group = []
        list_best = []
        for i in range(len(group)):
            sorted_team = sorted(group[i], key=lambda temp: (temp[self.ID_FIT], sum(temp[self.ID_POS_BIN])))
            sorted_group.append(sorted_team)
            list_best.append( deepcopy(sorted_team[self.ID_MIN_PROB]) )
        return sorted_group, list_best

    def _train__(self):
        pop, group = self._create_population__(self.n_clusters)
        pop, g_best = self._sort_pop_and_get_global_best__(pop, self.ID_FIT, self.ID_MIN_PROB)  # single element
        group, p_best = self._sort_group_and_get_best_solution_in_team(group)                   # multiple element

        # Loop iterations
        for epoch in range(self.epoch):
            print("Iteration_BHGSO: ", epoch)

            ## Loop based on the number of cluster in swarm (number of gases type)
            for i in range(self.n_clusters):

                ### Loop based on the number of individual in each gases type
                for j in range( self.n_elements):

                    F = -1.0 if uniform() < 0.5 else 1.0

                    ##### Based on Eq. 8, 9, 10
                    self.H_j = self.H_j * exp(-self.C_j * ( 1.0/exp(-epoch/self.epoch) - 1.0/self.T0 ))
                    S_ij = self.K * self.H_j * self.P_ij
                    gama = self.beta * exp(- ((p_best[i][self.ID_FIT] + self.epxilon) / (group[i][j][self.ID_FIT] + self.epxilon)))

                    X_ij = group[i][j][self.ID_POS] + F * uniform() * gama * (p_best[i][self.ID_POS] - group[i][j][self.ID_POS]) + \
                        F * uniform() * self.alpha * (S_ij * g_best[self.ID_POS] - group[i][j][self.ID_POS])

                    temp = self._to_binary_and_update_fit__(X_ij, group[i][j])
                    group[i][j] = temp
                    pop[i*self.n_elements + j] = temp

            ## Update Henry's coefficient using Eq.8
            self.H_j = self.H_j * exp(-self.C_j * (1.0 / exp(-epoch / self.epoch) - 1.0 / self.T0))
            ## Update the solubility of each gas using Eq.9
            S_ij = self.K * self.H_j * self.P_ij
            ## Rank and select the number of worst agents using Eq. 11
            N_w = int(self.pop_size * (uniform(0, 0.1) + 0.1))
            ## Update the position of the worst agents using Eq. 12
            sorted_id_pos = argsort([ x[self.ID_FIT] for x in pop ])

            for item in range(N_w):
                id = sorted_id_pos[item]
                j = id % self.n_elements
                i = int((id-j) / self.n_elements)
                X_new = self._create_solution__()
                pop[id] = X_new
                group[i][j] = X_new

            group, p_best = self._sort_group_and_get_best_solution_in_team(group)
            pop, g_best = self._sort_pop_and_update_global_best__(pop, self.ID_MIN_PROB, g_best)
            self.loss_train.append(g_best[self.ID_FIT])
            if self.log:
                print("> Epoch: {}, Best fit: {}".format(epoch + 1, g_best[self.ID_FIT]))
        return g_best[self.ID_POS_BIN], g_best[self.ID_FIT], self.loss_train, g_best[self.ID_kappa], g_best[self.ID_precision], g_best[self.ID_recall], g_best[self.ID_f1_score]



class ImprovedBinaryHGSO(Root):


    def __init__(self, objective_func=None, transfer_func=None, problem_size=1000, domain_range=(0, 1), log=True,
                 epoch=100, pop_size=10, lsa_epoch=10, seed_num=42, n_clusters = 2):
        Root.__init__(self, objective_func, transfer_func, problem_size, domain_range, log, lsa_epoch, seed_num)
        self.epoch = epoch
        self.pop_size = pop_size
        
        self.n_clusters = n_clusters
        self.n_elements = int(self.pop_size / self.n_clusters)

        self.T0 = 298.15
        self.K = 1.0
        self.beta = 1.0
        self.alpha = 1
        self.epxilon = 0.05

        self.l1 = 5E-3
        self.l2 = 1E+2
        self.l3 = 1E-2
        self.H_j = self.l1 * uniform()
        self.P_ij = self.l2 * uniform()
        self.C_j = self.l3 * uniform()
        self.weighting_factor = 0.8 
        self.crossover_rate = 0.9
    def _create_population__(self, n_clusters=0):
        pop = []
        group = []
        for i in range(n_clusters):
            team = []
            for j in range(self.n_elements):
                temp = self._create_solution__()
                team.append(temp)
                pop.append(temp)
            group.append(team)
        return pop, group

    def _sort_group_and_get_best_solution_in_team(self, group=None):
        sorted_group = []
        list_best = []
        for i in range(len(group)):
            sorted_team = sorted(group[i], key=lambda temp: (temp[self.ID_FIT], sum(temp[self.ID_POS_BIN])))
            sorted_group.append(sorted_team)
            list_best.append( deepcopy(sorted_team[self.ID_MIN_PROB]) )
        return sorted_group, list_best

    def _train__(self):
        pop, group = self._create_population__(self.n_clusters)
        pop, g_best = self._sort_pop_and_get_global_best__(pop=pop, id_fitness=self.ID_FIT, id_best=self.ID_MIN_PROB,apply_DE=True, apply_lsa=False,CR_Rate=self.crossover_rate,W_Factor=self.weighting_factor) 

        group, p_best = self._sort_group_and_get_best_solution_in_team(group)                   # multiple element

        # Loop iterations
        for epoch in range(self.epoch):
            print("Iteration_BHGSO: ", epoch)

            ## Loop based on the number of cluster in swarm (number of gases type)
            for i in range(self.n_clusters):

                ### Loop based on the number of individual in each gases type
                for j in range( self.n_elements):

                    F = -1.0 if uniform() < 0.5 else 1.0

                    ##### Based on Eq. 8, 9, 10
                    self.H_j = self.H_j * exp(-self.C_j * ( 1.0/exp(-epoch/self.epoch) - 1.0/self.T0 ))
                    S_ij = self.K * self.H_j * self.P_ij
                    gama = self.beta * exp(- ((p_best[i][self.ID_FIT] + self.epxilon) / (group[i][j][self.ID_FIT] + self.epxilon)))

                    X_ij = group[i][j][self.ID_POS] + F * uniform() * gama * (p_best[i][self.ID_POS] - group[i][j][self.ID_POS]) + \
                        F * uniform() * self.alpha * (S_ij * g_best[self.ID_POS] - group[i][j][self.ID_POS])

                    temp = self._to_binary_and_update_fit__(X_ij, group[i][j])
                    group[i][j] = temp
                    pop[i*self.n_elements + j] = temp

            ## Update Henry's coefficient using Eq.8
            self.H_j = self.H_j * exp(-self.C_j * (1.0 / exp(-epoch / self.epoch) - 1.0 / self.T0))
            ## Update the solubility of each gas using Eq.9
            S_ij = self.K * self.H_j * self.P_ij
            ## Rank and select the number of worst agents using Eq. 11
            N_w = int(self.pop_size * (uniform(0, 0.1) + 0.1))
            ## Update the position of the worst agents using Eq. 12
            sorted_id_pos = argsort([ x[self.ID_FIT] for x in pop ])

            for item in range(N_w):
                id = sorted_id_pos[item]
                j = id % self.n_elements
                i = int((id-j) / self.n_elements)
                X_new = self._create_solution__()
                pop[id] = X_new
                group[i][j] = X_new

            group, p_best = self._sort_group_and_get_best_solution_in_team(group)
            pop, g_best = self._sort_pop_and_update_global_best__(pop=pop, id_best=self.ID_MIN_PROB, g_best=g_best,apply_DE=True, apply_lsa=False,CR_Rate=self.crossover_rate,W_Factor=self.weighting_factor) 

            self.loss_train.append(g_best[self.ID_FIT])
            if self.log:
                print("> Epoch: {}, Best fit: {}".format(epoch + 1, g_best[self.ID_FIT]))
        return g_best[self.ID_POS_BIN], g_best[self.ID_FIT], self.loss_train, g_best[self.ID_kappa], g_best[self.ID_precision], g_best[self.ID_recall], g_best[self.ID_f1_score]

