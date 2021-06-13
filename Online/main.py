#!/usr/bin/env python
# coding: utf-8

# basic stuff
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
import matplotlib.pylab as plt

# import warnings filter
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)

# ML stuff
from sklearn.preprocessing import MaxAbsScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


# import Base
from Online.Utils import *
from Online.Parties import System, User, UserClear, MLE, MLEClear


if __name__ == '__main__':
    # ------------------------------------------------------------------------------
    # GET DATA HERE
    # U = get("U6")
    # questions = pd.DataFrame(get("questionsbias6"))
    # U_train, U_test, y_train, y_test = train_test_split(U, questions , test_size=0.9)
    # ------------------------------------------------------------------------------
   
    set1 = {"N": 256, "lr":1.2, "precision": 25, "ipb": 15, "k":128, "epoch": 15, "item": 2}
    problemset = set1

    #save scores
    loss = np.zeros(5, 4, problemset["epoch"], 3)
    name = "test_n{epoch}_q{item}_k{k}_p{precision}_ipb{ipb}_N{N}_lr{lr}".format(**problemset)


    # define system
    sys = System(problemset)
    paramset = sys.get_params()
    keyset = sys.get_public_key_set()

    for q in range(5):
        #create ML engines
        MLEngine = MLE(problemset, paramset, keyset) 
        MLEngineClear = MLEClear(problemset, paramset, keyset)

        # create Users
        users = list()
        usersClear = list()
        for ui,yi in zip(U_train,y_train.iloc[:,q]):
            users.append(User(problemset, paramset, keyset, ui, yi))
            usersClear.append(UserClear(problemset, paramset, keyset, ui, yi))

        # init new latent vector
        v, vclear = sys.newQuestion(6)

        # create new Item
        MLEngine.init_v(v, 7)
        MLEngineClear.init_v(vclear, 7)
        print("Initial vector vj:", vclear)

        #launch regression
        n_iter = 0
        gradvs = dict()
        gradvsclear= dict()

       
        while n_iter != problemset['epoch']:
            # 1. USER SIDE
            # create pool for multicore user evaluation
            arg = list()
            for i, u in enumerate(users):
                arg.append((u, v, i))
            with Pool(4) as p:
                results = p.map(helper, arg)
                for i, r in enumerate(results):
                    gradvs[i] = r
            # users in clear
            for i, uclear in enumerate(usersClear):        
                gradvsclear[i] = uclear.GD(vclear)

            
            #shuffle before making mini-batches
            gradvs, gradvsclear = shuffle(gradvs, gradvsclear, random_state=1) 

            # 2. MLE SIDE
            # MLE aggregates and updates gradients
            v, watch = MLEngine.GD(gradvs)
            vclear = MLEngineClear.GD(gradvsclear)


            # 3. Evaluation
            vdecrypted = sys.decrypt(v)

            loss[q][0][n_iter] = get_score(y_train.iloc[:,q], U_train, vclear[0:7])   
            loss[q][1][n_iter] = get_score(y_test.iloc[:,q], U_test, vclear[0:7])   
            loss[q][2][n_iter] = get_score(y_train.iloc[:,q], U_train, vdecrypted[0:7])   
            loss[q][3][n_iter] = get_score(y_test.iloc[:,q], U_test, vdecrypted[0:7]) 

            n_iter += 1


        plt.plot(loss[q, 0, 0:max_iter, 1])
        plt.plot(loss[q, 1, 0:max_iter, 1])
        plt.plot(loss[q, 2, 0:max_iter, 1])
        plt.plot(loss[q, 3, 0:max_iter, 1])

        plt.show()
