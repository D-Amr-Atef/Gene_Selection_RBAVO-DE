#
# ------------------------------------------------------------------------------------------------------%
# Created by "Ahmed Gad"                                                                                %
#                                                                                                       %
#       Email:      ahmed.gad_it@yahoo.com                                                              %
#       Homepage:   https://www.researchgate.net/profile/Ahmed_G_Gad                                    %
#       Github:     https://github.com/ahmedgad19891                                                    %
#-------------------------------------------------------------------------------------------------------%

class Settings:
    """ This is general of all Models """

    ID_MIN_PROB = 0 # min problem
    ID_MAX_PROB = -1    # max problem

    EPSILON = 10E-10
    
    def __init__(self, problem_size=1000, domain_range=(0, 1), log=True, epoch=100, pop_size=30):
        """
        Parameters
        ----------
        objective_func :
        problem_size :
        domain_range :
        log :
        """
        self.problem_size = problem_size
        self.domain_range = domain_range
        self.log = log
        self.epoch = epoch
        self.pop_size = pop_size

        self.OMEGA = 0.99    #weightage for accuracy and no. of features 

    def _train__(self):
        pass
