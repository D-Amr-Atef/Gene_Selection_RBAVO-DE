#
# ------------------------------------------------------------------------------------------------------%
# Created by "Ahmed Gad"                                                                                %
#                                                                                                       %
#       Email:      ahmed.gad_it@yahoo.com                                                              %
#       Homepage:   https://www.researchgate.net/profile/Ahmed_G_Gad                                    %
#       Github:     https://github.com/ahmedgad19891                                                    %
#-------------------------------------------------------------------------------------------------------%

from numpy import exp, sin, pi, array, mean, zeros, where
from numpy.random import uniform, randint
from numpy.linalg import norm
from copy import deepcopy
from models.root import Root

class BinaryASO(Root):
    """
    My binary version of Atom Search Optimization (ASO)
        https://doi.org/10.1016/j.knosys.2018.08.030
        https://www.mathworks.com/matlabcentral/fileexchange/67011-atom-search-optimization-aso-algorithm
    """
    ID_POS = 0      # current position
    ID_POS_BIN = 1  # current binary position
    ID_FIT = 2      # current fitness
    ID_VEL = 3      # Velocity
    ID_M = 4        # Mass of atom

    def __init__(self, objective_func=None, transfer_func=None, problem_size=1000, domain_range=(0, 1), log=True,
                 epoch=100, pop_size=10, lsa_epoch=10, seed_num=42, alpha=50, beta=0.2):
        Root.__init__(self, objective_func, transfer_func, problem_size, domain_range, log, lsa_epoch, seed_num)
        self.epoch = epoch
        self.pop_size = pop_size
        
        self.alpha = alpha                  # Depth weight
        self.beta = beta                    # Multiplier weight

    def _update_mass__(self, pop):
        pop = sorted(pop, key=lambda temp: (temp[self.ID_FIT], sum(temp[self.ID_POS_BIN])))
        best_fit = pop[0][self.ID_FIT]
        worst_fit = pop[-1][self.ID_FIT]
        sum_fit = sum([item[self.ID_FIT] for item in pop])
        for it in pop:
            it[self.ID_M] = exp( (it[self.ID_FIT] - best_fit)/(worst_fit - best_fit) ) / sum_fit
        return pop

    def _find_LJ_potential__(self, iteration, average_dist, radius):
        c = (1 - iteration / self.epoch) ** 3
        # g0 = 1.1, u = 2.4
        rsmin = 1.1 + 0.1 * sin((iteration+1) / self.epoch * pi / 2)
        rsmax = 1.24
        if radius/average_dist < rsmin:
            rs = rsmin
        else:
            if radius/average_dist > rsmax:
                rs = rsmax
            else:
                rs = radius / average_dist
        potential = c * (12 * (-rs)**(-13) - 6 * (-rs)**(-7))
        return potential

    def _acceleration__(self, pop, g_best, iteration):
        eps = 2**(-52)
        pop = self._update_mass__(pop)

        G = exp(-20.0 * (iteration+1) / self.epoch)
        k_best = int(self.pop_size - (self.pop_size - 2) * ((iteration + 1) / self.epoch) ** 0.5) + 1
        k_best_pop = deepcopy(sorted(pop, key=lambda it: it[self.ID_M], reverse=True)[:k_best])
        k_best_pos = [item[self.ID_POS] for item in k_best_pop]
        mk_average = mean(array(k_best_pos))

        acc_list = zeros((self.pop_size, self.problem_size))
        for i in range(0, self.pop_size):
            dist_average = norm(pop[i][self.ID_POS] - mk_average)
            temp = zeros((self.problem_size))

            for atom in k_best_pop:
                # calculate LJ-potential
                radius = norm(pop[i][self.ID_POS]-atom[self.ID_POS])
                potential = self._find_LJ_potential__(iteration, dist_average, radius)
                temp += potential * uniform(0, 1, self.problem_size) * \
                    ((atom[self.ID_POS]-pop[i][self.ID_POS])/(radius + eps))
            temp = self.alpha * temp + self.beta * (g_best[self.ID_POS] - pop[i][self.ID_POS])
            # calculate acceleration
            acc = G * temp / pop[i][self.ID_M]
            acc_list[i] = acc
        return acc_list


    def _train__(self):
        velocity = uniform(self.domain_range[0], self.domain_range[1], self.problem_size)
        mass = 0.0
        pop = [self._create_solution__()+[velocity]+[mass] for _ in range(self.pop_size)]
        pop, g_best = self._sort_pop_and_get_global_best__(pop, self.ID_FIT, self.ID_MIN_PROB)

        # Calculate acceleration.
        atom_acc_list = self._acceleration__(pop, g_best, iteration=0)

        for epoch in range(0, self.epoch):
            # Update velocity based on random dimensions and position of global best

            for i in range(0, self.pop_size):
                velocity_rand = uniform(self.domain_range[0], self.domain_range[1], self.problem_size)
                velocity = velocity_rand * pop[i][self.ID_VEL] + atom_acc_list[i]
                temp = pop[i][self.ID_POS] + velocity
                pop[i] = self._to_binary_and_update_fit__(temp, pop[i])

            current_best = self._get_global_best__(pop, self.ID_FIT, self.ID_MIN_PROB)
            if current_best[self.ID_FIT] < g_best[self.ID_FIT]:
                g_best = deepcopy(current_best)
            else:
                r = randint(0, self.pop_size)
                pop[r] = deepcopy(g_best)

            pop = sorted(pop, key=lambda temp: (temp[self.ID_FIT], sum(temp[self.ID_POS_BIN])))
            self.loss_train.append(g_best[self.ID_FIT])
            if self.log:
                print("> Epoch: {}, Best fit: {}".format(epoch + 1, g_best[self.ID_FIT]))
        return g_best[self.ID_POS_BIN], g_best[self.ID_FIT], self.loss_train, g_best[self.ID_kappa], g_best[self.ID_precision], g_best[self.ID_recall], g_best[self.ID_f1_score]




class ImprovedBinaryASO(Root):
  
  
    ID_POS = 0      # current position
    ID_POS_BIN = 1  # current binary position
    ID_FIT = 2      # current fitness
    ID_VEL = 3      # Velocity
    ID_M = 4        # Mass of atom

    def __init__(self, objective_func=None, transfer_func=None, problem_size=1000, domain_range=(0, 1), log=True,
                 epoch=100, pop_size=10, lsa_epoch=10, seed_num=42, alpha=50, beta=0.2):
        Root.__init__(self, objective_func, transfer_func, problem_size, domain_range, log, lsa_epoch, seed_num)
        self.epoch = epoch
        self.pop_size = pop_size
        
        self.alpha = alpha                  # Depth weight
        self.beta = beta                    # Multiplier weight
        self.weighting_factor = 0.8 
        self.crossover_rate = 0.9
        
    def _update_mass__(self, pop):
        pop = sorted(pop, key=lambda temp: (temp[self.ID_FIT], sum(temp[self.ID_POS_BIN])))
        best_fit = pop[0][self.ID_FIT]
        worst_fit = pop[-1][self.ID_FIT]
        sum_fit = sum([item[self.ID_FIT] for item in pop])
        for it in pop:
            it[self.ID_M] = exp( (it[self.ID_FIT] - best_fit)/(worst_fit - best_fit) ) / sum_fit
        return pop

    def _find_LJ_potential__(self, iteration, average_dist, radius):
        c = (1 - iteration / self.epoch) ** 3
        # g0 = 1.1, u = 2.4
        rsmin = 1.1 + 0.1 * sin((iteration+1) / self.epoch * pi / 2)
        rsmax = 1.24
        if radius/average_dist < rsmin:
            rs = rsmin
        else:
            if radius/average_dist > rsmax:
                rs = rsmax
            else:
                rs = radius / average_dist
        potential = c * (12 * (-rs)**(-13) - 6 * (-rs)**(-7))
        return potential

    def _acceleration__(self, pop, g_best, iteration):
        eps = 2**(-52)
        pop = self._update_mass__(pop)

        G = exp(-20.0 * (iteration+1) / self.epoch)
        k_best = int(self.pop_size - (self.pop_size - 2) * ((iteration + 1) / self.epoch) ** 0.5) + 1
        k_best_pop = deepcopy(sorted(pop, key=lambda it: it[self.ID_M], reverse=True)[:k_best])
        k_best_pos = [item[self.ID_POS] for item in k_best_pop]
        mk_average = mean(array(k_best_pos))

        acc_list = zeros((self.pop_size, self.problem_size))
        for i in range(0, self.pop_size):
            dist_average = norm(pop[i][self.ID_POS] - mk_average)
            temp = zeros((self.problem_size))

            for atom in k_best_pop:
                # calculate LJ-potential
                radius = norm(pop[i][self.ID_POS]-atom[self.ID_POS])
                potential = self._find_LJ_potential__(iteration, dist_average, radius)
                temp += potential * uniform(0, 1, self.problem_size) * \
                    ((atom[self.ID_POS]-pop[i][self.ID_POS])/(radius + eps))
            temp = self.alpha * temp + self.beta * (g_best[self.ID_POS] - pop[i][self.ID_POS])
            # calculate acceleration
            acc = G * temp / pop[i][self.ID_M]
            acc_list[i] = acc
        return acc_list


    def _train__(self):
        velocity = uniform(self.domain_range[0], self.domain_range[1], self.problem_size)
        mass = 0.0
        pop = [self._create_solution__()+[velocity]+[mass] for _ in range(self.pop_size)]
        pop, g_best = self._sort_pop_and_get_global_best__(pop=pop, id_fitness=self.ID_FIT, id_best=self.ID_MIN_PROB,apply_DE=True, apply_lsa=False,CR_Rate=self.crossover_rate,W_Factor=self.weighting_factor) 

        # Calculate acceleration.
        atom_acc_list = self._acceleration__(pop, g_best, iteration=0)

        for epoch in range(0, self.epoch):
            # Update velocity based on random dimensions and position of global best

            for i in range(0, self.pop_size):
                velocity_rand = uniform(self.domain_range[0], self.domain_range[1], self.problem_size)
                velocity = velocity_rand * pop[i][self.ID_VEL] + atom_acc_list[i]
                temp = pop[i][self.ID_POS] + velocity
                pop[i] = self._to_binary_and_update_fit__(temp, pop[i])

            current_best = self._get_global_best__(pop, self.ID_FIT, self.ID_MIN_PROB)
            if current_best[self.ID_FIT] < g_best[self.ID_FIT]:
                g_best = deepcopy(current_best)
            else:
                r = randint(0, self.pop_size)
                pop[r] = deepcopy(g_best)

            pop = sorted(pop, key=lambda temp: (temp[self.ID_FIT], sum(temp[self.ID_POS_BIN])))
            self.loss_train.append(g_best[self.ID_FIT])
            if self.log:
                print("> Epoch: {}, Best fit: {}".format(epoch + 1, g_best[self.ID_FIT]))
        return g_best[self.ID_POS_BIN], g_best[self.ID_FIT], self.loss_train, g_best[self.ID_kappa], g_best[self.ID_precision], g_best[self.ID_recall], g_best[self.ID_f1_score]
