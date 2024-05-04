from numpy.random import uniform, randint,permutation,rand,standard_normal
from numpy import abs,zeros,ones, clip,log,arange,floor,sum,square,sqrt,eye,matmul,power,diag,argsort,sort,array,matlib,triu,exp,linalg
from models.root import Root
# from mealpy.root import Root
from copy import deepcopy

 

class BaseWDO(Root):

    """
    Basic-Classic Wind Driven Optimization (WDO)
    """
    ID_POS = 0      # current position
    ID_POS_BIN = 1  # current binary solution
    ID_FIT = 2      # current fitness
    ID_VEL = 3      # Velocity
     
#---------------------------------------------------------------------------     
    
    def __init__(self, objective_func=None, transfer_func=None, problem_size=50, domain_range=(-1, 1), log=True,
                  epoch=750, pop_size=100, lsa_epoch=10, seed_num=42, RT=3, g=0.2, alp=0.4, c=0.4, max_v=0.3):

        Root.__init__(self, objective_func, transfer_func, problem_size, domain_range, log, lsa_epoch, seed_num)
  
         
        self.epoch = epoch
        self.pop_size = pop_size
        self.RT = RT                # RT coefficient
        self.g = g                  # gravitational constant
        self.alp = alp              # constants in the update equation
        self.c = c                  # coriolis effect
        self.max_v = max_v          # maximum allowed speed
      

#---------------------------------------------------------------------------
    def _train__(self):
        """
        # pop is the set of "air parcel" - "solution"
        # air parcel: is the set of gas atoms . Each atom represents a dimension in solution and has its own velocity
        # pressure represented by fitness value
        """
        pop = [self._create_solution_WDO__() for _ in range(self.pop_size)]
        # print ('vel:',pop[0][:])
        # g_best = self._get_global_best__(pop, self.ID_FIT, self.ID_MIN_PROB)
        # pop, g_best = self._sort_pop_and_get_global_best__(pop=pop, id_fitness=self.ID_FIT, id_best=self.ID_MIN_PROB, apply_lsa=False)
        pop, g_best = self._sort_pop_and_get_global_best_WDO__(pop=pop, id_fitness=self.ID_FIT, id_best=self.ID_MIN_PROB,apply_COSAM=False)

        for epoch in range(1,self.epoch):

            # Update velocity based on random dimensions and position of global best
            for i in range(self.pop_size):

                #------------------------------------------------------
                rand_dim = permutation(range(0,self.problem_size)) #random perm   

                # rand_dim = randint(0, self.problem_size)
                
                # velocity influence from another dimnsion
                temp = pop[i][self.ID_VEL][rand_dim] * ones(self.problem_size)
                
                #------------------------------------------------------
                vel = (1 - self.alp)*pop[i][self.ID_VEL] - self.g * pop[i][self.ID_POS] + \
                      abs(1-(1.0/(i+1))) * self.RT * (g_best[self.ID_POS] - pop[i][self.ID_POS]) + self.c * temp / (i+1)
                   
         
                #------------------------------------------------------    
                #check velocity
                # Given an interval, values outside the interval are clipped to the interval edges
                #  [-0.3,0.3]                       
                vel = clip(vel, -self.max_v, self.max_v)

                # Update air parcel positions, check the bound and calculate pressure (fitness)
                pos = pop[i][self.ID_POS] + vel
                
                # Given an interval, values outside the interval are clipped to the interval edges
                # [-1,1]:(domain range)
                pos = self._amend_solution_faster__(pos)
                
                #------------------------------------------------------
                # x_new = ((self.dimMax-self.dimMin) * (pos+1)/2) + self.dimMin

                # pop[i] = self._to_binary_and_update_fit_WDO__(pos,vel, pop[i])
              
                # x_new_bin = self._to_binary__(x_new, pop[i][self.ID_POS_BIN])
                solution_new_bin = self._to_binary__(pos, pop[i][self.ID_POS_BIN])
                fit_new = self.OMEGA if sum(solution_new_bin) == 0 else self._fitness_model__(solution_new_bin)
                pop[i] = [pos,solution_new_bin,fit_new, vel]

                ###################################################################
                # Fitness value        

                # fitness = self.OMEGA if sum(x_new_bin) == 0 else self._fitness_model__(x_new_bin)

                #------------------------------------------------------
                # fit = self._fitness_model__(pos)
                # pop[i] = [pos,solution_new_bin,fit_new, vel]
              
            #------------------------------------------------------
            #------------------------------------------------------
            # g_best = self._update_global_best__(pop, self.ID_MIN_PROB, g_best)
            # pop, g_best = self._sort_pop_and_update_global_best__(pop=pop, id_best=self.ID_MIN_PROB, g_best=g_best, apply_lsa=False)
            pop, g_best = self._sort_pop_and_update_global_best_WDO__(pop=pop, id_best=self.ID_MIN_PROB, g_best=g_best,apply_COSAM=False)
            self.loss_train.append(g_best[self.ID_FIT])
            if self.log:
                print("> Epoch: {}, Best fit: {}".format(epoch + 1, g_best[self.ID_FIT]))
                
        #------------------------------------------------------
        #------------------------------------------------------
        # return g_best[self.ID_POS], g_best[self.ID_FIT], self.loss_train
        return g_best[self.ID_POS_BIN], g_best[self.ID_FIT], self.loss_train


########################################################################################################
########################################################################################################
########################################################################################################

class AdaptiveWDO(BaseWDO):

    """
    Adaptive Wind Driven Optimization (AdaptiveWDO)
    """
    
    
    def __init__(self, objective_func=None, transfer_func=None, problem_size=50, domain_range=(-1, 1), log=True,
                  epoch=750, pop_size=100, lsa_epoch=10, seed_num=42, max_v=0.3):

        Root.__init__(self, objective_func, transfer_func, problem_size, domain_range, log, lsa_epoch, seed_num)
  
          
        self.epoch = epoch
        self.pop_size = pop_size
        
        ################################################################
        self.rec = {'arx': rand(4,self.pop_size) }
        #---------------------
        self.alp = self.rec['arx'][0,:]              # RT coefficient
        self.g = self.rec['arx'][1,:]                # gravitational constant
        self.c = self.rec['arx'][2,:]                # constants in the update equation
        self.RT = self.rec['arx'][3,:]               # coriolis effect
        ################################################################
      
        self.max_v = max_v                           # maximum allowed speed
      

#'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
#'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

    def _cmaes__(self,counteval=None, rec=None, npop=None, pres=None, dim=None): 
        # Refer to purecmaes.m -- https://www.lri.fr/~hansen/purecmaes.m
        # 
        # counteval -- Iteration counter from WDO.
        # rec -- Record of prior values used in CMAES.
        # npop -- number of population members from WDO, each member gets their own set of coefficients determined by the CMAES.
        # pres -- pressure(cost function) computed by WDO for the set of coefficients that CMEAS picked last iteration
        # dim -- number of dimensions of CMAES optimization
        
        if counteval==1:   #Initialization step
            # define 'records' dictionary to keep track the CMAES values over iterations. 
            #print('Init Stage')
            rec['N'] = dim
            rec['xmean'] = rand(dim,1)
            rec['sigma'] = 0.5
            rec['lambda'] = npop
            rec['mu'] = npop/2
            rec['weights'] = log((npop/2)+1/2) - log(arange(1,floor(npop/2)+1))
            rec['mu'] = floor(rec['mu']).astype(int)
            rec['weights'] = rec['weights']/sum(rec['weights'])
            rec['mueff'] = square(sum(rec['weights'])) / sum( rec['weights'] * rec['weights'])
            rec['cc'] = (4+rec['mueff']/rec['N']) / (rec['N']+4 +2*rec['mueff']/rec['N'])
            rec['cs'] = (rec['mueff']+2) / (rec['N'] + rec['mueff']+5)
            rec['c1'] = 2 / ((square(rec['N']+1.3)) + rec['mueff'])
            rec['cmu'] = min(1-rec['c1'], 2*(rec['mueff']-2+1/rec['mueff'])/(square(rec['N']+2)+rec['mueff']))
            rec['damps'] = 1 + 2*max(0, sqrt((rec['mueff'] -1)/(rec['N']+1))-1) + rec['cs']
            rec['pc'] = zeros(dim)
            rec['ps'] = zeros(dim)
            rec['B'] = eye(dim,dim)
            rec['D'] = ones(dim)
            rec['C'] = matmul(  matmul( rec['B'], diag(power(rec['D'],2)) ) , rec['B'].T  )
            rec['invsqrtC'] = matmul( matmul(rec['B'], diag(power(rec['D'],-1))) , rec['B']) 
            rec['eigeneval'] = 0
            rec['chiN'] = power(rec['N'],0.5)* (1-1/(4+rec['N'])+1/(21*square(rec['N'])) )
            
         
        #get fitness from WDO pressure
        rec['arfitness'] = pres
          
        
        # sort fitness and compute weighted mean into xmean
        arindex = argsort(pres)
        
        rec['arindex'] = arindex
        rec['arfitness'] = sort(pres)
        rec['xold'] = rec['xmean']
        mu = rec['mu']
        ridx = arindex[0:mu.astype(int)]
        recarx = array(rec['arx'])
        rec['xmean'] = matmul(  recarx[:,ridx], rec['weights'] ).reshape(dim,1)            
        
        rec['ps'] = (1-rec['cs']) * rec['ps'] + sqrt(rec['cs']*(2-rec['cs'])*rec['mueff'])  * matmul(rec['invsqrtC'] , (rec['xmean']-rec['xold'])).T / rec['sigma']
        rec['hsig'] = int( sum(rec['ps']*rec['ps']) / (1-power((1-rec['cs']),(2*counteval/rec['lambda']))) / rec['N']  <  2+(4/(rec['N']+1))   )
        rec['pc'] = (1-rec['cc']) * rec['pc'] + rec['hsig'] * sqrt(rec['cc']*(2-rec['cc'])*rec['mueff']) * (rec['xmean']-rec['xold']).T / rec['sigma']       
        rec['artmp'] = (1/rec['sigma']) * (recarx[:,ridx]) -matlib.repmat(rec['xold'],1,rec['mu'])
        
        
        
        
        rec['C'] = (1-rec['c1']-rec['cmu']) * rec['C']+ rec['c1'] * (rec['pc'] * rec['pc'].T + (1-rec['hsig']) * rec['cc']*(2-rec['cc']) * rec['C'])\
                     + rec['cmu'] * matmul( matmul(rec['artmp'] , diag(rec['weights'])) , rec['artmp'].T)
        
        rec['sigma'] = rec['sigma']*exp( (rec['cs']/rec['damps'])*(linalg.norm(rec['ps'])/rec['chiN']-1) )
        
        if (counteval-rec['eigeneval']) >  (rec['lambda'] / (rec['c1']+rec['cmu'])/rec['N']/10):
            rec['eigeneval'] = counteval
            rec['C'] = triu(rec['C']) + triu(rec['C'],1).T
            rec['D'], rec['B'] = linalg.eigh(rec['C'])
            rec['D'] = sqrt(rec['D'])
            rec['invsqrtC'] = matmul( matmul(rec['B'], diag( rec['D']**(-1)) )  , rec['B'].T)
            
            
        for k in range(1,rec['lambda']):
            recarx[:,k] = rec['xmean'].T + (rec['sigma']* matmul(rec['B'], ((rec['D']*( standard_normal(size=(rec['N'],1))).T)).reshape(dim,1) )).T
        rec['arx'] = recarx
        
        return rec

    
#---------------------------------------------------------------------------
    def _train__(self):
        """
        # pop is the set of "air parcel" - "solution"
        # air parcel: is the set of gas atoms . Each atom represents a dimension in solution and has its own velocity
        # pressure represented by fitness value
        """
        pop = [self._create_solution_WDO__() for _ in range(self.pop_size)]
        pop_Fitness=zeros(self.pop_size)
        # g_best = self._get_global_best__(pop, self.ID_FIT, self.ID_MIN_PROB)
        # pop, g_best = self._sort_pop_and_get_global_best__(pop=pop, id_fitness=self.ID_FIT, id_best=self.ID_MIN_PROB, apply_lsa=False)
        pop, g_best = self._sort_pop_and_get_global_best_WDO__(pop=pop, id_fitness=self.ID_FIT, id_best=self.ID_MIN_PROB,apply_COSAM=False)

        for epoch in range(1,self.epoch):

            # Update velocity based on random dimensions and position of global best
            for i in range(self.pop_size):

                #------------------------------------------------------
                rand_dim = permutation(range(0,self.problem_size)) #random perm   

                # rand_dim = randint(0, self.problem_size)
                
                # velocity influence from another dimnsion
                temp = pop[i][self.ID_VEL][rand_dim] * ones(self.problem_size)
                
                #------------------------------------------------------
                vel = (1 - self.alp[i])*pop[i][self.ID_VEL] - self.g[i] * pop[i][self.ID_POS] + \
                      abs(1-(1.0/(i+1))) * self.RT[i] * (g_best[self.ID_POS] - pop[i][self.ID_POS]) + self.c[i] * temp / (i+1)
                            
                #------------------------------------------------------    
                #check velocity
                # Given an interval, values outside the interval are clipped to the interval edges
                #  [-0.3,0.3]                       
                vel = clip(vel, -self.max_v, self.max_v)

                # Update air parcel positions, check the bound and calculate pressure (fitness)
                pos = pop[i][self.ID_POS] + vel
                
                # Given an interval, values outside the interval are clipped to the interval edges
                # [-1,1]:(domain range)
                pos = self._amend_solution_faster__(pos)
                
                #------------------------------------------------------
                # x_new = ((self.dimMax-self.dimMin) * (pos+1)/2) + self.dimMin

                # pop[i] = self._to_binary_and_update_fit_WDO__(pos,vel, pop[i])
                
                solution_new_bin = self._to_binary__(pos, pop[i][self.ID_POS_BIN])
                fit_new = self.OMEGA if sum(solution_new_bin) == 0 else self._fitness_model__(solution_new_bin)
                pop[i] = [pos,solution_new_bin,fit_new, vel]              
                
                pop_Fitness[i]=deepcopy(pop[i][self.ID_FIT])                            

                # x_new_bin = self._to_binary__(x_new, pop[i][self.ID_POS_BIN])

                ###################################################################
                # Fitness value        

                # fitness = self.OMEGA if sum(x_new_bin) == 0 else self._fitness_model__(x_new_bin)

                #------------------------------------------------------
                # fit = self._fitness_model__(pos)
                # pop[i] = [pos,x_new,x_new_bin, fitness, vel]
              
            #------------------------------------------------------
            #------------------------------------------------------
            # g_best = self._update_global_best__(pop, self.ID_MIN_PROB, g_best)
            # pop, g_best = self._sort_pop_and_update_global_best__(pop=pop, id_best=self.ID_MIN_PROB, g_best=g_best, apply_lsa=False)
            pop, g_best = self._sort_pop_and_update_global_best_WDO__(pop=pop, id_best=self.ID_MIN_PROB, g_best=g_best,apply_COSAM=False) 
            self.loss_train.append(g_best[self.ID_FIT])
            if self.log:
                print("> Epoch: {}, Best fit: {}".format(epoch + 1, g_best[self.ID_FIT]))
            #------------------------------------------------------
            #------------------------------------------------------
          
            #update inherent parameters through cmaes:
            self.rec = self._cmaes__(epoch,self.rec,self.pop_size,pop_Fitness,self.rec['arx'].shape[0])
               
            #------------------------------------------------------
            self.alp = self.rec['arx'][0,:]
            self.g = self.rec['arx'][1,:]
            self.c = self.rec['arx'][2,:]
            self.RT = self.rec['arx'][3,:]
                 
                
                    
            
        #------------------------------------------------------
        #------------------------------------------------------
        # return g_best[self.ID_POS], g_best[self.ID_FIT], self.loss_train
        return g_best[self.ID_POS_BIN], g_best[self.ID_FIT], self.loss_train





########################################################################################################
########################################################################################################
########################################################################################################

class ImprovedAdaptiveWDO(AdaptiveWDO):

    """
    My WDO-Improved Adaptive Wind Driven Optimization (ImprovedAdaptiveWDO)
    """
    
    def __init__(self, objective_func=None, transfer_func=None, problem_size=50, domain_range=(-1, 1), log=True,
                  epoch=750, pop_size=100, lsa_epoch=10, seed_num=42, max_v=0.3):

        Root.__init__(self, objective_func, transfer_func, problem_size, domain_range, log, lsa_epoch, seed_num)
  
          
        self.epoch = epoch
        self.pop_size = pop_size
        
        ################################################################
        self.rec = {'arx': rand(4,self.pop_size) }
        #---------------------
        self.alp = self.rec['arx'][0,:]              # RT coefficient
        self.g = self.rec['arx'][1,:]                # gravitational constant
        self.c = self.rec['arx'][2,:]                # constants in the update equation
        self.RT = self.rec['arx'][3,:]               # coriolis effect
        ################################################################
      
        self.max_v = max_v                           # maximum allowed speed
      


    
#---------------------------------------------------------------------------
    def _train__(self):
        """
        # pop is the set of "air parcel" - "solution"
        # air parcel: is the set of gas atoms . Each atom represents a dimension in solution and has its own velocity
        # pressure represented by fitness value
        """
        pop = [self._create_solution_WDO__() for _ in range(self.pop_size)]
        pop_Fitness=zeros(self.pop_size)
        # g_best = self._get_global_best__(pop, self.ID_FIT, self.ID_MIN_PROB)
        
        # pop, g_best = self._sort_pop_and_get_global_best__(pop=pop, id_fitness=self.ID_FIT, id_best=self.ID_MIN_PROB, apply_lsa=False)
        
        #------------------------------------------------------
        # New Modify (Cross Over & Simulated Annealing with Two Phase Mutation)
        pop, g_best = self._sort_pop_and_get_global_best_WDO__(pop=pop, id_fitness=self.ID_FIT, id_best=self.ID_MIN_PROB,apply_COSAM=True,Start_Temp=self.ID_Start_T,Final_Temp=self.ID_Final_T,Cool_Rate=self.ID_Cool_R)
        #------------------------------------------------------
        
        # pop, g_best= self._apply_COSAM__(pop=pop, g_best=g_best, apply_simann=True,Start_Temp=self.ID_Start_T,Final_Temp=self.ID_Final_T,Cool_Rate=self.ID_Cool_R)
        # print(g_best[self.ID_POS_BIN])
        
        
        

        for epoch in range(1,self.epoch):

            # print("    Iteration: ", epoch)

            # Update velocity based on random dimensions and position of global best
            for i in range(self.pop_size):

                #------------------------------------------------------
                rand_dim = permutation(range(0,self.problem_size)) #random perm   

                # rand_dim = randint(0, self.problem_size)
                
                # velocity influence from another dimnsion
                temp = pop[i][self.ID_VEL][rand_dim] * ones(self.problem_size)
                
                #------------------------------------------------------
                vel = (1 - self.alp[i])*pop[i][self.ID_VEL] - self.g[i] * pop[i][self.ID_POS] + \
                      abs(1-(1.0/(i+1))) * self.RT[i] * (g_best[self.ID_POS] - pop[i][self.ID_POS]) + self.c[i] * temp / (i+1)
                            
                #------------------------------------------------------    
                #check velocity
                # Given an interval, values outside the interval are clipped to the interval edges
                #  [-0.3,0.3]                       
                vel = clip(vel, -self.max_v, self.max_v)

                # Update air parcel positions, check the bound and calculate pressure (fitness)
                pos = pop[i][self.ID_POS] + vel
                
                
                # Given an interval, values outside the interval are clipped to the interval edges
                # [-1,1]:(domain range)
                # New Modify (_amend_solution_random_faster__ 3RA)
                pos = self._amend_solution_random_faster__(pos)
                
                #------------------------------------------------------
                # x_new = ((self.dimMax-self.dimMin) * (pos+1)/2) + self.dimMin
                # New Modify (_to_binary_and_update_fit_WDO__)
                pop[i] = self._to_binary_and_update_fit_WDO__(pos,vel, pop[i])
                
                pop_Fitness[i]=deepcopy(pop[i][self.ID_FIT])                            

                # x_new_bin = self._to_binary__(x_new, pop[i][self.ID_POS_BIN])

                ###################################################################
                # Fitness value        

                # fitness = self.OMEGA if sum(x_new_bin) == 0 else self._fitness_model__(x_new_bin)

                #------------------------------------------------------
                # fit = self._fitness_model__(pos)
                # pop[i] = [pos,x_new,x_new_bin, fitness, vel]
              
            #------------------------------------------------------
            #------------------------------------------------------
            # g_best = self._update_global_best__(pop, self.ID_MIN_PROB, g_best)
            
            # pop, g_best = self._sort_pop_and_update_global_best__(pop=pop, id_best=self.ID_MIN_PROB, g_best=g_best, apply_lsa=False)
            
            
            # print("  Iteration:", epoch)
            # print("Before:" , g_best[self.ID_POS_BIN],(g_best[self.ID_FIT]))

            #------------------------------------------------------
            # New Modify (Cross Over & Simulated Annealing with Two Phase Mutation)
            pop, g_best = self._sort_pop_and_update_global_best_WDO__(pop=pop, id_best=self.ID_MIN_PROB, g_best=g_best,apply_COSAM=True,Start_Temp=self.ID_Start_T,Final_Temp=self.ID_Final_T,Cool_Rate=self.ID_Cool_R)
            #------------------------------------------------------
            
            # pop, g_best= self._apply_COSAM__(pop=pop, g_best=g_best, apply_simann=True,Start_Temp=self.ID_Start_T,Final_Temp=self.ID_Final_T,Cool_Rate=self.ID_Cool_R)

            # print("After:" , g_best[self.ID_POS_BIN],(g_best[self.ID_FIT]))

            self.loss_train.append(g_best[self.ID_FIT])
            if self.log:
                
                ("> Epoch: {}, Best fit: {}".format(epoch + 1, g_best[self.ID_FIT]))
            #------------------------------------------------------
            #------------------------------------------------------
          
            #update inherent parameters through cmaes:            
            self.rec = self._cmaes__(epoch,self.rec,self.pop_size,pop_Fitness,self.rec['arx'].shape[0])
               
            #------------------------------------------------------
            self.alp = self.rec['arx'][0,:]
            self.g = self.rec['arx'][1,:]
            self.c = self.rec['arx'][2,:]
            self.RT = self.rec['arx'][3,:]
            
   
             
        #------------------------------------------------------
        #------------------------------------------------------
        # return g_best[self.ID_POS], g_best[self.ID_FIT], self.loss_train
        return g_best[self.ID_POS_BIN], g_best[self.ID_FIT], self.loss_train

