#basic stuff
import time
import pandas as pd
import numpy as np
from math import log2

#crypto stuff
from ckks.ckks_decryptor import CKKSDecryptor
from ckks.ckks_encoder import CKKSEncoder
from ckks.ckks_encryptor import CKKSEncryptor
from ckks.ckks_evaluator import CKKSEvaluator
from ckks.ckks_key_generator import CKKSKeyGenerator
from ckks.ckks_parameters import CKKSParameters

#ML stuff
from sklearn.model_selection import KFold
from Offline.Utils import *


class Base:
    # def __init__(self):
    
    #force vector v to be of length of ciphertext
    def prepare(self, v):
        b = np.zeros(int(self.params.poly_degree/2))
        b[0:len(v)] = v
        return list(b)

    #grab some info about the ciphertext (for debugging)
    def info(self, ct):
        return (ct.modulus.bit_length(), ct.scaling_factor.bit_length(), sys.decrypt(ct))

# ------------------------
# - User Class -----------
# ------------------------
class User(Base):
    def __init__(self, pk, params, u, r):
        #Paillier parameters
        self.pk = pk
        self.params = params
        
        self.encoder = CKKSEncoder(params)
        self.encryptor = CKKSEncryptor(params, self.pk)
        self.evaluator = CKKSEvaluator(params) 
        
        # make personal vector p
        self.u = np.insert(u, 0, 1.) # having u[0] = 1 for intercept
        self.uy = np.insert(self.u, len(self.u), r) #add the preference r
        self.uy = self.prepare(self.uy)
        # self.M = np.outer(self.u, uy).T
        
        #encoding and encrypting p
        self.Upt = self.encoder.encode(self.uy, self.params.scaling_factor)
        self.Uct = self.encryptor.encrypt(self.Upt)
         
    
    # Gradient Descent on User side    
    def GD(self):
        # nothing to do here
        return 

# ------------------------
# - ML Engine Class ------
# ------------------------
class MLE (Base): 
    def __init__(self, keys, params, problemparams):
        #save keys
        self.keys = keys
        self.params = params
        
        self.encoder = CKKSEncoder(params)
        self.encryptor = CKKSEncryptor(params, self.keys["pk"])
        self.evaluator = CKKSEvaluator(params)
        
        #ML parameters
        self.lr = problemparams["lr"]
        self.d = problemparams["d"]
        self.batch_size = int(problemparams["N"]/(2*self.d))
        self.itp = problemparams["ipb"]

        
    def init_v(self, v, k):
        self.k = k
        self.v = v
        self.n_iter_ = 1 
    
    def upload(self, Ms):
        self.Ms = Ms
        self.num_users = len(Ms)
        self.nbatches = int(len(self.Ms)/self.batch_size)
    
    # concat b users in a single ciphertext [[U]]
    def make_random_batches(self):
        kf = KFold(n_splits=self.nbatches, shuffle=True)
        self.Mopt = list()

        #make random splits
        for train_index, test_index in kf.split(range(len(self.Ms))):
            # here we start a new ct [[U]]
            tmp = self.Ms[test_index[0]]
            for j in range(1, self.batch_size):
                tmp = self.evaluator.rotate(tmp, self.d, self.keys["rot_keys"][self.d])
                tmp = self.evaluator.add(tmp, self.Ms[test_index[j]])
            self.Mopt.append(tmp)

    # Gradient Descent on Interm side
    def GD(self, max_iters, sys):
        self.time = np.zeros((max_iters, 5, self.nbatches))
        vs = list()

        # generate a unique mask containing the learning rate 
        mask = np.insert(np.zeros((self.batch_size,self.d-1)), 0, [np.ones(self.batch_size)], axis=1).ravel()
        mask *= (self.lr/len(self.Ms))
        self.maskpt = self.encoder.encode(list(mask), self.params.scaling_factor)

        for self.n_iter_ in range(1,max_iters+1):

            # STEP 3a : shuffle and make batches
            self.make_random_batches()

            # STEPS 3b-d
            v = self.make_epoch()

            #STEP 3e: bootstrapping
            if(self.n_iter_%self.itp ==0):
                print("Bootstrapping ...")
                oldv, self.v = self.evaluator.bootstrap(self.v, self.rot_keys, self.conj_key, self.relin_key, self.encoder)

        return vs

    # sumNaive as in pdf
    def sum(self, A, dim=8, coef=1):
        tmp = A
        for i in range(int(log2(dim))):
            step = coef * 2 ** i
            rot = self.evaluator.rotate(tmp, step, self.keys["rot_keys"][step])
            tmp = self.evaluator.add(tmp, rot)
        return tmp

    # extract and distribute sum
    def distribute(self, Edirty):

        #remove dirty parts of the sum
        Eclean = self.evaluator.multiply_plain(Edirty, self.maskpt)
        Eclean = self.evaluator.rescale(Eclean, self.params.scaling_factor) 

        #distribute the sum (also on y-slot)
        tmp = Eclean
        for i in range(int(log2(self.d))): 
            step = 2 ** i
            rot = self.evaluator.rotate(tmp, step, self.keys["rot_keys"][step])
            tmp = self.evaluator.add(tmp, rot)

        step = self.d*(self.batch_size-1)+1
        tmp = self.evaluator.rotate(tmp, step, self.keys["rot_keys"][step])

        # set y-slot to 0 
        step = self.d*(self.batch_size-1)+1
        rot = self.evaluator.rotate(Eclean, step, self.keys["rot_keys"][step])
        tmp = self.evaluator.subtract(tmp, rot)
        return tmp

    def make_epoch(self):
        
        #parse mini batches
        for i in range(len(self.Mopt)):  
            start = time.time()
            U = self.evaluator.lower_modulus(self.Mopt[i], self.params.scaling_factor ** ((3*i)))#+((self.n_iter_-1)%self.itp * 3 *len(self.Mopt)))) 
            
            # STEP 3b : compute E = U * W
            start = time.time()
            E = self.evaluator.multiply(U, self.v, self.keys["relin_key"])
            E = self.evaluator.rescale(E, self.params.scaling_factor) 
            
            # sum columns 
            E_sum_dirty = self.sum(E, dim=self.d, coef=1)

            # clean sum & distribute
            E_sum = self.distribute(E_sum_dirty)
            self.time[self.n_iter_-1][1][i] += time.time()-start 

            # STEP 3c : grad = E * U
            start = time.time()
            U = self.evaluator.lower_modulus(U, self.params.scaling_factor ** 2)
            grad = self.evaluator.multiply(E_sum, U, self.keys["relin_key"])
            grad = self.evaluator.rescale(grad, self.params.scaling_factor) 

            # sum rows
            grad_sum = self.sum(grad, dim=self.batch_size, coef=self.d)
            self.time[self.n_iter_-1][2][i] += time.time()-start 

            # STEP 3d : parameter update
            start = time.time()
            self.v = self.evaluator.lower_modulus(self.v, self.params.scaling_factor ** 3)
            self.v = self.evaluator.subtract(self.v, grad_sum)
            self.time[self.n_iter_-1][3][i] += time.time()-start 
        
        
        return self.v


# ----------------------------
# - ML Engine Class (clear) --
# ----------------------------
class MLEClear(MLE):
    def __init__(self, keys, params, problemparams):
        super().__init__(keys, params, problemparams)

    def concat(self):
        self.nbatches = int(len(self.Ms)/self.batch_size)

        self.Mopt = list()
        for i in range(self.nbatches):
            # here we start a new ct
            tmp = self.Ms[i*self.batch_size]
            for j in range(1, self.batch_size):
                tmp = np.roll(tmp, self.r) #rotate
                tmp = tmp + self.Ms[i*self.batch_size + j]
            self.Mopt.append(tmp)  

     # Gradient Descent on Interm side
    def GD(self, max_iters):
        vs = list()
        for self.n_iter_ in range(1,max_iters+1):
            self.concat()
            v = self.make_epoch()
            vs.append(v)
        return vs

    def sum(self, A, dim=8, coef=1):
        tmp = A
        for i in range(int(log2(dim))):
            step = (1) * coef * 2 ** i
            rot = np.roll(tmp, step)
            tmp = tmp + rot
        tmp = np.roll(tmp, -(dim*coef)+coef)
        return tmp
    
    def distribute(self, Edirty):
        #remove dirty parts of the sum
        mask = np.insert(np.zeros((self.batch_size,self.r-1)), 0, [np.ones(self.batch_size)], axis=1).ravel()
        mask *= 1
        Eclean = Edirty * mask

        #distribute the sum (also on y-slot)
        tmp = Eclean
        for i in range(int(log2(self.r))): 
            step = 2 ** i
            rot = np.roll(tmp, step)
            tmp = tmp + rot

        # set y-slot to 0 
        step = self.r-1
        rot = np.roll(Eclean, step)
        tmp = tmp - rot
        return tmp

    def make_epoch(self):
        #define lr (for whole iter)
        lr = self.lr/len(self.Ms) 
        lr = np.zeros(int(self.params.poly_degree/2))+lr

        #parse mini batches
        for i in range(self.nbatches):  
            U = self.Mopt[i].ravel()

            # compute E = U * W
            E = U * self.v 

            # sum columns 
            E_sum_dirty = self.sum(E, dim=self.r, coef=1)

            # clean & distribute sum
            E_sum = self.distribute(E_sum_dirty)

            # grad = E * U
            grad = E_sum * U 

            # sum rows
            grad_sum = self.sum(grad, dim=self.batch_size, coef=self.r)

            diff = grad_sum * lr
            self.v = self.v - diff
            
        return self.v   

# ----------------------------
# - System Class -------------
# ----------------------------
class System(Base): 
    def __init__(self, problemparams):
        self.batch_size = int(problemparams["N"]/(2*problemparams["d"]))
        self.max_users = problemparams["k"]
        self.iters_per_bootstrapping = problemparams["ipb"]
        self.d = problemparams["d"]

        poly_degree = problemparams["N"]
        delta = problemparams["precision"]
        delta_0 = 10
        taylor=7

        L = 3 * int(self.max_users/self.batch_size)
        Q0 = delta * L * problemparams["ipb"] + delta + delta_0

        ciph_modulus = 1 << Q0 
        big_modulus = 1 << Q0 + (8+taylor)*(delta+delta_0) #+ int(22*precision) + 10
        self.scaling_factor = 1 << delta

        params = CKKSParameters(poly_degree=poly_degree,
                                ciph_modulus=ciph_modulus,
                                big_modulus=big_modulus,
                                scaling_factor=self.scaling_factor,
                                taylor_iterations=taylor, 
                                prime_size=None)
        
        # KeyGen
        key_generator = CKKSKeyGenerator(params)
        self.pk = key_generator.public_key
        self.sk = key_generator.secret_key
        self.relin_key = key_generator.relin_key
        self.conj_key = key_generator.generate_conj_key()
        self.rot_keys = {}
        for i in range(int(poly_degree/2)):
            self.rot_keys[i] = key_generator.generate_rot_key(i)
        
        # encoder generators
        self.encoder = CKKSEncoder(params)
        self.encryptor = CKKSEncryptor(params, self.pk)
        self.decryptor = CKKSDecryptor(params, self.sk)
        self.evaluator = CKKSEvaluator(params)

        # parameters
        self.params = params


    def get_params(self):
        return self.params
    def get_public_key_set(self):
        return {"pk": self.pk, "relin_key":self.relin_key, "conj_key": self.conj_key, "rot_keys": self.rot_keys}


    # define the initial random values of the new latent vector v_j
    def newQuestion(self, k):
        v_rand = [np.random.rand(k)]*self.batch_size
        v_rand = np.insert(v_rand, k, [np.ones(self.batch_size)*(-1)], axis=1)
        v = np.insert(v_rand, 0, [np.zeros(self.batch_size)], axis=1).ravel()
        
        pl = self.encoder.encode(self.prepare(v), self.scaling_factor)
        vct = self.encryptor.encrypt(pl)
        
        return vct, v    
    
    def getPk(self):
        return self.pk
    
    def decrypt(self, v):
        pl = self.decryptor.decrypt(v)
        clear = self.encoder.decode(pl)
        return [x.real for x in clear]
    
    def decryptNorm(self, v):
        pl = self.decryptor.decrypt(v)
        clear = self.encoder.decode(pl)
        return [x.real+x.imag for x in clear]
