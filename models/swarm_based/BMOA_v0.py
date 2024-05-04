from models.OptimizationAlgorithm import OptimizationAlgorithm
from math import gamma
from numpy.random import uniform, normal, choice
from numpy import sin, abs, pi, ones, clip
from copy import deepcopy


# FIXME
class BMOA_v0(OptimizationAlgorithm):
    ID_POS = 0  # current position
    ID_POS_BIN = 1  # current binary position
    ID_FIT = 2  # current fitness

    def __init__(
        self,
        objective_func=None,
        transfer_func=None,
        problem_size=1000,
        domain_range=(-1, 1),
        log=True,
        epoch=100,
        pop_size=10,
        lsa_epoch=10,
        seed_num=42,
        P=0.5,
    ):
        OptimizationAlgorithm.__init__(
            self,
            objective_func,
            transfer_func,
            problem_size,
            domain_range,
            log,
            lsa_epoch,
            seed_num,
        )
        self.epoch = epoch
        self.pop_size = pop_size
        self.P = P
        """meerkat individuals hunt or search for food with the same probability
        """
        self.r = 1

        self.sentry = 0.3
        """behavior based on the presence or absence of predators and alarms, 
        under the safe condition of rand < sentry (default is 0.3),
        """

    def _train__(self):
        # Step 1: Initialize MOA’s population 𝑋 using Eq. (1).
        pop = [self._create_solution__() for _ in range(self.pop_size)]
        g_best = self._get_global_best__(pop, self.ID_FIT, self.ID_MIN_PROB)
        # Step 2: while Termination conditions are not met
        FEs = 0
        while FEs < self.epoch:
            Terminated = False
            # Step 3: Record the initial position of each individual using Eq. (2).
            """meerkats diffused their search outwards based on the initial position Direct,
            searching for food and observing natural enemies."""
            direct = deepcopy(pop)

            # Step 4: Calculate 𝑠𝑡𝑒𝑝 using Eq. (3).
            step = 2 - 2 * FEs / (self.epoch - 1)

            # Step 5: for Each meerkat in the population
            for i in range(self.pop_size):
                # Step 6: emergencies, meerkats scout for natural predators
                # or disasters and issue warnings for individual meerkats to
                # take refuge or fight back, as follows.
                # Hunting activities and vigilance
                if uniform() < self.sentry:
                    # Step 7: searching for food and observing natural enemies
                    if self.P > uniform():
                        # Update the current solution using Eq. (4).
                        pop[i][self.ID_POS] = (
                            pop[i][self.ID_POS] + step * direct[i][self.ID_POS]
                        )
                    # Step 9:
                    else:
                        #  the meerkat randomly finds other companions during
                        #  its search and approaches them for coordinated hunting
                        # Step 10: Randomly select an individual other than itself as 𝑋𝑗 .
                        j = self.random_choice_except(self.pop_size, i)
                        # Step 11: Update the current solution using Eq. (5).
                        pop[i][self.ID_POS] = pop[i][self.ID_POS] + step * (
                            pop[j][self.ID_POS]
                            - (uniform() + 0.5) * pop[i][self.ID_POS]
                        )
                # Step : 12 flee or fight against the enemy
                else:
                    # meerkats to gather in the direction of the leader
                    # Step 13: Calculate x_emergency using Eq. (7).
                    x_emergency = pop[i][self.ID_POS] + (
                        2 * uniform() * g_best[self.ID_POS] - pop[i][self.ID_POS]
                    )
                    # Border check and rebirth
                    x_emergency = self._fix_boundaries__(x_emergency)
                    x_emergency = self._to_binary_and_update_fit__(pop[i], x_emergency)
                    FEs = FEs + 1
                    if FEs == self.epoch:
                        Terminated = True
                        break
                    self.loss_train.append(g_best[self.ID_FIT])
                    # Step 14: if Fitnessx_emergencyss<Fitness(xxi𝑖𝑡𝑛𝑒𝑠𝑠𝑖𝑡𝑛𝑒𝑠𝑠(𝑋i) then
                    if x_emergency[self.ID_FIT] < pop[i][self.ID_FIT]:
                        # Step 15: Update the new position 𝑋 with Eq. (8).
                        pop[i] = x_emergency
                    # Step 16: Else
                    else:
                        # Meerkats have low fitness toward the leader,
                        # indicating that when they find that
                        # invaders block the road or the enemy is
                        # too strong, the meerkats will
                        # flee in the opposite direction for asylum
                        # Step 17 Update the new position 𝑋 with Eq. (6), (9).
                        div = uniform() + 0.1
                        x_emergency = div * pop[i][self.ID_POS] - (
                            2 * uniform() * g_best[self.ID_POS] - pop[i][self.ID_POS]
                        )
                        # Border check and rebirth
                        x_emergency = self._fix_boundaries__(x_emergency)
                        pop[i] = self._to_binary_and_update_fit__(pop[i], x_emergency)
                        FEs = FEs + 1
                        if FEs == self.epoch:
                            Terminated = True
                            break
                        self.loss_train.append(g_best[self.ID_FIT])
                g_best = self._update_global_best__(pop, self.ID_MIN_PROB, g_best)
            if Terminated:
                break
            if self.log:
                print("> Epoch: {}, Best fit: {}".format(FEs, g_best[self.ID_FIT]))

            # Random direction exploration
            # Step 18: if 𝑟𝑎𝑛𝑑 < 𝑃 then
            if uniform() < self.P:
                # Step 19: for Each meerkat in the population do
                for i in range(self.pop_size):
                    # Step 20: Random update the new position of 𝑋 using Eq. (10).
                    pop[i][self.ID_POS] = (
                        pop[i][self.ID_POS]
                        + (2 * uniform() - 1)
                        * (
                            pop[i][self.ID_POS]
                            + uniform() * self.levy(self.problem_size)
                        )
                        * step
                    )
            for i in range(self.pop_size):
                pop[i][self.ID_POS] = self._fix_boundaries__(pop[i][self.ID_POS])
            # Step 21: Border check and rebirth using Eq. (11).
        # Step 22: return The best solution and its position

        self.loss_train.append(g_best[self.ID_FIT])
        return g_best[self.ID_POS_BIN], g_best[self.ID_FIT], self.loss_train

    def _create_solution__(self):
        # eq. 1
        pos = (
            normal(0.5, 0.3, self.problem_size)
            * (self.domain_range[1] - self.domain_range[0])
            + self.domain_range[0]
        )
        pos_bin = self._to_binary__(pos, ones(self.problem_size))
        fit = self.OMEGA if sum(pos_bin) == 0 else self._fitness_model__(pos_bin)
        return [pos, pos_bin, fit]

    def random_choice_except(self, a: int, excluding: int, size=None, replace=True):
        # generate random values in the range [0, a-1)
        choices = choice(a - 1, size, replace=replace)
        # shift values to avoid the excluded number
        return choices + (choices >= excluding)

    def levy(self, n, beta=1.6):
        num = gamma(1 + beta) * sin(pi * beta / 2)
        den = gamma((1 + beta) / 2) * beta * (2 ** ((beta - 1) / 2))
        sigma_u = (num / den) ** (1 / beta)
        sigma_v = 1
        u = normal(0, sigma_u, n)
        v = normal(0, sigma_v, n)
        S = u / (abs(v) ** (1 / beta))
        return S

    def Check_Dead_Or_Rebirth(self, x, lb, ub):
        """Reference
        @INPROCEEDINGS{7257135,
        author={Juárez-Castillo, Efrén and Pérez-Castro, Nancy and Mezura-Montes, Efrén},
        booktitle={2015 IEEE Congress on Evolutionary Computation (CEC)},
        title={A novel boundary constraint-handling technique for constrained numerical optimization problems},
        year={2015},
        volume={},
        number={},
        pages={2034-2041},
        doi={10.1109/CEC.2015.7257135}}

        Args:
            x (_type_): _description_
            lb (_type_): _description_
            ub (_type_): _description_
            gbest (_type_): _description_

        Returns:
            _type_: _description_
        """
        dim = len(x)
        new_x = deepcopy(x)
        for i in range(dim):
            if new_x[i] < lb:
                new_x[i] = lb + (ub - lb) * normal(0.5, 0.3)
            if new_x[i] > ub:
                new_x[i] = lb + (ub - lb) * normal(0.5, 0.3)

        return new_x

    def _fix_boundaries__(self, X):
        return clip(X, self.domain_range[0], self.domain_range[1])
