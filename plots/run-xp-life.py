#!/usr/bin/env python3

from graphTools import *
from expTools import *
import os

# Dictionnaire avec les options de compilations d'apres commande
options = {}
options["-k"] = ["life"]
options["-i"] = [200]
options["-v"] = ["mpi_col"] # "ompfor", "ompcol2"
options["-wt"] = ["AVX"]
options["-s"] = [1024]
# options["-ts"] = [2**i for i in range(1, 9)]
options["-tw"] = [2**i for i in range(1, 11)]
options["-th"] = [2**i for i in range(1, 11)]
options["-a"] = ["moultdiehard2474"]
#options["-mpi"] = ["\"--oversubscribe --mca btl tcp,self -hostfile my_machines --map-by node -np 4\""]
options["-mpi"] = ["\"-np 4\""]


# Pour renseigner l'option '-of' il faut donner le chemin depuis le fichier easypap
options["-of"] = ["./data/perf/MPI1.csv"]



# Dictionnaire avec les options OMP
ompenv = {}
ompenv["OMP_SCHEDULE"]=["static"]
ompenv["OMP_NUM_THREADS"] = [1] + list(range(2, 9, 2))
#ompenv["OMP_PLACES"] = ["cores"]k,,,

nbruns = 1

# Lancement des experiences
execute('./run ', ompenv, options, nbruns, verbose=True, easyPath=".")

# # Lancement de la version seq avec le nombre de thread impose a 1
# options["-v"] = ["seq"]
# ompenv["OMP_NUM_THREADS"] = [1]
# execute('./run', ompenv, options, nbruns, verbose=False, easyPath=".")
