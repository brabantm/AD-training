import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

from Offline.Utils import *
from Offline.Parties import System, User, MLE, MLEClear



# ------------------------------------------------------------------------------
# GET DATA HERE
# U = get("U6")
# questions = pd.DataFrame(get("questionsbias6"))
# U_train, U_test, y_train, y_test = train_test_split(U, questions , test_size=0.9)
# ------------------------------------------------------------------------------

set1 = {"N": 64, "lr":1.2, "precision": 25, "ipb": 15, "d":8, "k":128, "epoch": 15, "item": 2}
problemset = set1

if __name__ == "__main__":
    k = problemset["k"]
    d = problemset["d"]
    
    # create System
    sys = System(problemset)
    keys = sys.get_public_key_set()
    params = sys.get_params()

    # create new item
    v, vclear = sys.newQuestion(problemset["d"]-2) # minus 2 for item bias and preference
    q = problemset["item"]

    # create the k users
    users = [User(keys['pk'], params, U_train[i,:], y_train.iloc[i,q]) for i in range(problemset["k"])]

    #create MLE
    MLEngine = MLE(keys, params, problemset)
    MLEngine.init_v(v, 8)
    #create MLE in clear for comparison
    MLEngineClear = MLEClear(keys, params, problemset)
    MLEngineClear.init_v(vclear, 8)

    # upload all user data
    MLEngine.upload([u.Uct for u in users])
    MLEngineClear.upload([u.uy for u in users])
    
    # Gradient descent
    vs = MLEngine.GD(problemset["epoch"], sys)
    vs_clear = MLEngineClear.GD(problemset["epoch"])

    # save results
    m = np.zeros((problemset["epoch"],4))
    for i in range(len(vs_clear)):

        pred = U_test[0:k,0:d].dot(np.array(vs[i][1:d+1])) + vs[i][0]
        m[i][0] = mean_squared_error(y_test.iloc[0:k,q].values, pred, squared=False)

        predclear = U_test[0:k,0:d].dot(np.array(vs_clear[i][1:d+1])) + vs_clear[i][0]
        m[i][1] = mean_squared_error(y_test.iloc[0:k,q].values, predclear, squared=False)

        pred = U_train[0:k,0:d].dot(np.array(vs[i][1:d+1])) + vs[i][0]
        m[i][2] = mean_squared_error(y_train.iloc[0:k,q].values, pred, squared=False)

        predclear = U_train[0:k,0:d].dot(np.array(vs_clear[i][1:d+1])) + vs_clear[i][0]
        m[i][3] = mean_squared_error(y_train.iloc[0:k,q].values, predclear, squared=False)

    #plot results
    plt.plot(m)
    plt.show()
  
