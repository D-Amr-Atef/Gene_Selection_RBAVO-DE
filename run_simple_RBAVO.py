#
#-------------------------------------------------------------------------------------------------------%
from models.swarm_based.BSSA import BinarySSA, ImprovedBinarySSA
from models.swarm_based.BABC import BinaryABC
from models.swarm_based.BPSO import BinaryPSO
from models.swarm_based.BBA  import BinaryBA, ImprovedBinaryBA
from models.swarm_based.BGWO import BinaryGWO
from models.swarm_based.BWOA import BinaryWOA
from models.swarm_based.BGOA import BinaryGOA
from models.swarm_based.BSFO import BinarySFO, ImprovedBinarySFO
from models.swarm_based.BHHO import BinaryHHO, ImprovedBinaryHHO
from models.swarm_based.BBSA import BinaryBSA
from models.swarm_based.BMOA import BinaryMOA,ImprovedBinaryMOA
from models.swarm_based.BAO import BinaryAO,ImprovedBinaryAO
from models.swarm_based.BAVO import BinaryAVO,ImprovedBinaryAVO
from models.swarm_based.BBBO import BinaryBBO,ImprovedBinaryBBO
from models.physics_based.BASO import BinaryASO, ImprovedBinaryASO
from models.physics_based.BHGSO import BinaryHGSO, ImprovedBinaryHGSO
# from models.evolutionary_based.DE import BaseDE
# from models.swarm_based.BAVO import ImprovedBinaryAVO
#-------------------------------------------------------------------------------------------------------%
#-------------------------------------------------------------------------------------------------------%
from models.physics_based.BNRO import ImprovedBinaryNRO
from os import getcwd, path, makedirs,remove

#-------------------------------------------------------------------------------------------------------%
#-------------------------------------------------------------------------------------------------------%


from functions.fitness.ml.knn import KNN
from functions.fitness.ml.svm import SVM
# from functions.fitness.ml.mlp import MLP
from functions.fitness.ml.gnb import GNB
from functions.fitness.ml.dt import DT
# from functions.fitness.ml.cnn import CNN
from functions.fitness.ml.xgbtree import xgbTree
# from functions.fitness.ml.mlp import MLP
from functions.fitness.ml.random_forest import RandomForest


from functions.transfer.transfer_function import *

from evaluation.algorithm_evaluation import Evaluation




######################################################################################
#-------------------------------------------------------------------------------------------------------%
# RNA DataSets
dataset_list = ["BLCA","CESC","CHOL","COAD","ESCA","GBM","HNSC","KICH","KIRC","KIRP","LIHC","LUAD","LUSC","PAAD","PCPG","READ","SARC","SKCM","STAD","THCA","THYM","UCEC"]
#-------------------------------------------------------------------------------------------------------%
######################################################################################



######################################################################################
#-------------------------------------------------------------------------------------------------------%
# Algorithm: Nuclear Reaction Processes (ImprovedBinaryNRO)

# algo_dicts = {"RBNRO-DE": ImprovedBinaryNRO ,"BSSA": BinarySSA, "BABC": BinaryABC, "BPSO": BinaryPSO, "BBA": BinaryBA, "BGWO": BinaryGWO, "BWOA": BinaryWOA, "BGOA": BinaryGOA,
#                 "BSFO": BinarySFO, "BHHO": BinaryHHO, "BBSA": BinaryBSA, "BASO": BinaryASO, "BHGSO": BinaryHGSO}
algo_dicts = {"RBAVO-DE": ImprovedBinaryAVO}
######################################################################################
# Recent Algorithms
# algo_dicts = {"RBNRO-DE": ImprovedBinaryNRO, "BMOA": BinaryMOA, "BBBO": BinaryBBO, "BAO": BinaryAO, "BAVO": BinaryAVO} 

######################################################################################
# Hybrid Algorithms with RF and DE
# algo_dicts = {"RBNRO-DE": ImprovedBinaryNRO, "RBMOA-DE": ImprovedBinaryMOA, "RBBBO-DE": ImprovedBinaryBBO, "RBAO-DE": ImprovedBinaryAO, "RBAVO-DE": ImprovedBinaryAVO, "RBSSA-DE": ImprovedBinarySSA,
#               "RBSFO-DE": ImprovedBinarySFO, "RBHHO-DE": ImprovedBinaryHHO, "RBASO-DE": ImprovedBinaryASO, "RBHGSO-DE": ImprovedBinaryHGSO, "RBBA-DE": ImprovedBinaryBA} 
#-------------------------------------------------------------------------------------------------------%
######################################################################################

obj_fun_dicts = {"ML": [KNN, SVM]}
# obj_fun_dicts = {"ML": [KNN, SVM]}

# obj_fun_dicts = {"ML": [KNN]}

# trans_fun_dicts = {"S-Shaped": [s_v1, s_v1c, s_v2, s_v3, s_v4],
#                     "V-Shaped": [v_v1, v_v2, v_v3, v_v4]}
trans_fun_dicts = {"V-Shaped": [v_v4]}

# trans_fun_dicts = {"S-Shaped": [s_v1, s_v4],
#                    "V-Shaped": [v_v1, v_v2, v_v4]}



           
           
    
######################################################################################
## Setting parameters
num_runs = 30

domain_range = (-1, 1)
log = False

epoch = 100
pop_size = 10


## Local search algoithm
lsa_epoch = 20
param_result_path = getcwd() + "/Final_BNRO/Final_BNRO_result.doc"

# feat_combination= [500, 1000, 2000, 3000, 5000, 6000, 7000, 9000, 11000, 15000, 16000, 17000]
feat_combination = 500
######################################################################################
######################################################################################
# Run model
# for f_combina in feat_combination: 
if  path.exists(param_result_path):
        param_file=open(param_result_path,'a')
        param_file.write("\nNumber of Selected Features from Relief: {} \n\n".format(feat_combination))
        param_file.close()
              
md = Evaluation(dataset_list, algo_dicts, obj_fun_dicts, trans_fun_dicts, num_runs, domain_range, log, epoch, pop_size, lsa_epoch, feat_combination)
md._eval__()




