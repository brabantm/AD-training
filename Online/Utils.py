import numpy as np
import pandas as pd
from sklearn.preprocessing import MaxAbsScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pylab as plt
import pickle


import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
from tensorflow.keras.regularizers import l2, l1
from tensorflow.keras import layers, Model
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam

# Helper Functions

def get_score(real, X,v):
    pred = np.insert(X[:, :], 0, np.ones(len(X)), axis=1).dot(v)
    rmse = mean_squared_error(real, pred, squared=False)
    mae = mean_absolute_error(real, pred)
    r2 = r2_score(real, pred) 
    return r2, rmse, mae

def helper(t):
    (u,vclear, i) = t
    return u.GD(vclear)

def dump(obj, filename):
    with open(filename+".pickle", 'wb') as config_dictionary_file:
        pickle.dump(obj, config_dictionary_file)

def get(filename):
    a = open(filename + ".pickle", "rb")
    print("Unpickling...")
    return pickle.load(a)

def mse(true, predicted):
    """
    A function to compute the total mean square error
    """
   
    error = 0
    for i in range(len(predicted)):
        error += pow(true[i] - predicted[i], 2)
    return np.sqrt(error/len(predicted))

def parseData(data):
    subtables = ["election", "info", "questionsRaw", "weights", "questionsInfo"]
    columns = [['voterID', 'electionID', 'sourceTYPE', 'source','recTIME','recTYPE','questTYPE', 'soc_completion','N_answers','quest_completion'],
            ['districtID', 'language',  'birthYEAR','age','gender','zip','education','interest','position','pref_party'],
            ['language',  'birthYEAR', 'gender', 'zip', 'education', 'interest', 'position','pref_party'], [range(74,126)]]

    df = dict()
    print("init length", data.shape)

    dataNaN = data.replace(-9, np.nan)

    noQuestionNanIndex = dataNaN.iloc[:,20:73].dropna().index #.iloc[;,20:73]
    print("no NaN length", len(noQuestionNanIndex))

    df["election"] = data.iloc[noQuestionNanIndex][columns[0]]
    df["info"] = data.iloc[noQuestionNanIndex][columns[1]].replace([-9, -1, -1977, -990], np.nan)
    df["questionsRaw"] = data.iloc[noQuestionNanIndex,20:73]
    df["weights"] = data.iloc[noQuestionNanIndex,74:127]
    df["questionsInfo"] = pd.DataFrame()
    for sub in subtables:
        print("size", sub, df[sub].shape)
        
    df["questions"] = MaxAbsScaler().fit_transform(pd.concat([df["questionsRaw"], df["info"]], axis=1)) #MaxAbsScaler().fit_transform(
    df["questions"] = pd.DataFrame(df["questions"], columns=np.concatenate((df["questionsRaw"].columns,df["info"].columns)))
    return df


def plot_metric(h, metric):
    h = h.history
    train_metrics = h[metric]
    val_metrics = h['val_'+metric]
    epochs = range(1, len(train_metrics) + 1)
    plt.plot(epochs, train_metrics)
    plt.plot(epochs, val_metrics)
    plt.title('Training and validation '+ metric)
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend(["train_"+metric, 'val_'+metric])
    plt.show()

def makeForKeras(df):
    
    d = df.copy()
    cols = np.arange(0,len(df.iloc[0,:]))
    d.columns= cols
    d.insert(0, "index", np.arange(0,len(df)))
    d = pd.melt(d, id_vars=['index'], value_vars=cols)
    d.variable = d.variable.astype(int)
    return d

class EmbeddingLayer:
    def __init__(self, n_items, n_factors, name="", reg=l2(1e-5)):
        self.n_items = n_items
        self.n_factors = n_factors
        self.name = name
        self.reg = reg
    
    def __call__(self, x):
        x = layers.Embedding(self.n_items, self.n_factors, name=self.name,
                      embeddings_regularizer=self.reg)(x)
        x = layers.Reshape((self.n_factors,))(x)
        return x


def MF(df, n_factors, epoch, reg):
    n_users, n_items = df.shape
    train = makeForKeras(df)
    
    user = layers.Input(shape=(1,))
    u = EmbeddingLayer(n_users, n_factors, name='U', reg=l2(1e-4))(user)
    ub = EmbeddingLayer(n_users, 1, name='U-bias', reg=l1(0))(user)
    
    items = layers.Input(shape=(1,))
    m = EmbeddingLayer(n_items, n_factors, name="V", reg=l2(1e-4))(items)
    mb = EmbeddingLayer(n_items, 1, name="V-bias", reg=l1(reg))(items)
    x = layers.Dot(axes=1, activity_regularizer=None)([u, m])
    x = layers.Add()([x, ub, mb])
    model = Model(inputs=[user, items], outputs=x)
    opt = Adam(0.075, beta_1=0.9, beta_2=0.999)
    model.compile(optimizer=opt, loss='mean_squared_error', metrics=['mae'])
#     return model

    bs = int(len(train)/6)
    history = model.fit([train["index"].values, train.variable.values], train.value.values,batch_size=bs, epochs=epoch, verbose=0)
    #         plot_metric(history.history, "mse")
    U, V = model.get_layer(name='U').get_weights()[0], model.get_layer(name='V').get_weights()[0]
    Ubias = model.get_layer(name='U-bias').get_weights()[0]
    Vbias = model.get_layer(name='V-bias').get_weights()[0]
    
    return U, V, Ubias, Vbias
