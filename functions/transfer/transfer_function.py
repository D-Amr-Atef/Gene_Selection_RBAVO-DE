#
# ------------------------------------------------------------------------------------------------------%
# Created by "Ahmed Gad"                                                                                %
#                                                                                                       %
#       Email:      ahmed.gad_it@yahoo.com                                                              %
#       Homepage:   https://www.researchgate.net/profile/Ahmed_G_Gad                                    %
#       Github:     https://github.com/ahmedgad19891                                                    %
#-------------------------------------------------------------------------------------------------------%

from numpy import sqrt, pi, tanh, arctan
from scipy.special import expit
from math import erf
from numpy.random import uniform

## S-shaped transfer functions
def s_v1(gamma=None):     #convert to probability
        return expit(-gamma)

def s_v1c(gamma=None):
        return expit(gamma)

def s_v2(gamma=None):
        return expit(-gamma/2)
                
def s_v3(gamma=None):
        return expit(-gamma/3)

def s_v4(gamma=None):
        return expit(-gamma*2)


## V-shaped transfer functions
def v_v1(gamma=None):
        return abs(tanh(gamma))

def v_v2(gamma=None):
        val = (pi)**(0.5)
        val /= 2
        val *= gamma
        val = erf(val)
        return abs(val)

def v_v3(gamma=None):
        val = 1 + gamma*gamma
        val = sqrt(val)
        val = gamma/val
        return abs(val)

def v_v4(gamma=None):
        val=(pi/2)*gamma
        val=arctan(val)
        val=(2/pi)*val
        return abs(val)

if __name__ == "__main__":
    print(Transfer.sigmoid1(uniform()))    

