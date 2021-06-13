
#!/usr/bin/env python
# coding: utf-8

#basic stuff
from math import log2
import numpy as np
import pandas as pd
import time

#crypto stuff
from ckks.ckks_decryptor import CKKSDecryptor
from ckks.ckks_encoder import CKKSEncoder
from ckks.ckks_encryptor import CKKSEncryptor
from ckks.ckks_evaluator import CKKSEvaluator
from ckks.ckks_key_generator import CKKSKeyGenerator
from ckks.ckks_parameters import CKKSParameters

class Base:
    # def __init__(self):

    def prepare(self, v):
        b = np.zeros(int(self.params.poly_degree/2))
        b[0:len(v)] = v
        return list(b)


# ------------------------
# - User Class -----------
# ------------------------
class User (Base):
    def __init__(self,problemset, paramset, keyset, u, r):
        #set keys
        self.keys = keyset

        #set CKKS params
        self.params = paramset
        self.encoder = CKKSEncoder(paramset)
        self.encryptor = CKKSEncryptor(paramset, self.keys["pk"])
        self.evaluator = CKKSEvaluator(paramset)
        
        #ML parameters
        self.lr = problemset["lr"]/problemset["k"]

        #data
        self.u = np.insert(u, 0, 1) # having u[0] = 1 for intercept
        self.upt = self.encoder.encode(self.prepare(self.u), self.params.scaling_factor)
        self.ulrpt = self.encoder.encode(self.prepare(self.u*self.lr), self.params.scaling_factor)  #encode (u*lr)

        self.r = r
        self.rpt = self.encoder.encode(self.prepare(np.zeros(len(self.u))-r), self.params.scaling_factor)
    
    # Naive Sum. See Appendix A
    def sum(self, A, dim=8, coef=1):
        tmp = A
        for i in range(int(log2(dim))):
            step = coef * 2 ** i
            rot = self.evaluator.rotate(tmp, step, self.keys["rot_keys"][step])
            tmp = self.evaluator.add(tmp, rot)
        return tmp

    # Gradient Descent on User side    
    def GD(self, v):
        # Step 2b : local computation of the gradients
        dot = self.evaluator.multiply_plain(v, self.upt)
        dot = self.evaluator.rescale(dot, self.params.scaling_factor)
        pred = self.sum(dot) #sum columns

        e = self.evaluator.add_plain(pred, self.rpt) # add (-r) prediction
        
        gradv = self.evaluator.multiply_plain(e, self.ulrpt) #multiply with (lr*u)
        gradv = self.evaluator.rescale(gradv, self.params.scaling_factor)

        return gradv # add value for stopping criterion

# ------------------------
# - User (clear) Class ---
# ------------------------
class UserClear (User):
    def __init__(self,pk, rot_keys, params, u, r): #pk, rot_keys, params,
        super().__init__(pk, rot_keys, params, u, r)
        
    def GD(self, vclear, i=0):
        predclear = vclear.dot(self.u)
        eclear = - self.r + predclear
        gradvclear = eclear * (self.lr*self.u)

        return gradvclear

# ------------------------
# - MLE Class ------------
# ------------------------
class MLE (Base): 
    def __init__(self, problemset, paramset, keyset):
        #set keys
        self.keys = keyset
        
        #set CKKS params
        self.params = paramset
        self.encoder = CKKSEncoder(paramset)
        self.encryptor = CKKSEncryptor(paramset, self.keys["pk"])
        self.evaluator = CKKSEvaluator(paramset)
        
       

        #ML parameters
        self.lr = problemset["lr"]/problemset["k"]
        self.num_users = problemset["k"]
        self.batch_size = problemset["k"]
        self.iters_per_bootstrapping = problemset["ipb"] 
        self.nbatches = int(self.num_users/self.batch_size)

    # Save initial v sent by System   
    def init_v(self, v, k):
        self.v = v
        self.k = k
        self.n_iter_ = 1
    # Gradient Descent on MLE side
    def GD(self, gradvs):
        watch = dict() #timer

        #scale modulus only once
        self.v = self.evaluator.lower_modulus(self.v, self.params.scaling_factor**2)

        #parse mini batches (traditionnally only 1)
        for i in range(self.nbatches):   
            start = i*self.batch_size
            stop = start + self.batch_size
            
            #step 3c: Aggregate gradients of mini-batch
            t = time.time()
            gradv = gradvs[start] # prevent 1 additional addition
            for j in range(start+1, stop):
                gradv = self.evaluator.add(gradv, gradvs[j])
            watch["agg"] = time.time()-t

            #step 3d: Param update
            start = time.time()
            self.v = self.evaluator.subtract(self.v, gradv)
            watch["subtract"] = time.time()-start


        watch["bootstrap"] = 0. #if no bootstrapping this iteration

        #step 3e: Bootstrapping
        if(self.n_iter_%self.iters_per_bootstrapping == 0): #check if bootstrapping is needed
            start = time.time()
            oldv, self.v = self.evaluator.bootstrap(self.v, self.keys["rot_keys"], self.keys["conj_key"], self.keys["relin_key"], self.encoder)
            watch["bootstrap"] = time.time()-start
            print("bootstrap in", watch["bootstrap"])

        self.n_iter_ += 1

        return self.v, watch
# ------------------------
# - MLE (clear) Class ----
# ------------------------
class MLEClear(MLE):
    def __init__(self, problemset, paramset, keyset):
        super().__init__(problemset, paramset, keyset)
        
     # Gradient Descent on MLE side
    def GD(self, gradvsclear):
        
        #parse mini batches
        batches = int(self.num_users/self.batch_size)
        for i in range(batches): 
            start = i*self.batch_size
            stop = start + self.batch_size
            
            gradvclear = gradvsclear[start]
            for i in range(start+1, stop):
                gradvclear = gradvclear + gradvsclear[i]

            diffclear = gradvclear
            self.v -= diffclear
            
        self.n_iter_ += 1

        return self.v

# ------------------------
# - System Class ---------
# ------------------------
class System (Base): 
    def __init__(self, set):
        #set params
        self.iters_per_bootstrapping = set["ipb"]
        self.max_users = set["k"]
        taylor=7

        #scaling factors
        delta_0 = 10
        delta =  set["precision"]

        #depth of this protocol
        L = 2 * self.iters_per_bootstrapping 

        #small modulus
        Q0 = L*delta + delta_0 + delta
        ciph_modulus = 1 << Q0
        big_modulus = 1 << Q0 + (8+taylor)*(delta+delta_0)
        self.scaling_factor = 1 << delta
        
        #param generation
        self.params = CKKSParameters(poly_degree=set["N"],
                                ciph_modulus=ciph_modulus,
                                big_modulus=big_modulus,
                                scaling_factor=self.scaling_factor,
                                taylor_iterations=taylor,
                                prime_size=None)
        key_generator = CKKSKeyGenerator(self.params)
        
        #key generation
        self.pk = key_generator.public_key
        self.sk = key_generator.secret_key
        self.relin_key = key_generator.relin_key
        self.conj_key = key_generator.generate_conj_key()
        self.rot_keys = {}
        for i in range(int(self.params.poly_degree/2)):
            self.rot_keys[i] = key_generator.generate_rot_key(i)
        
        #generate encoder and decryptor
        dec = CKKSDecryptor(self.params, self.sk)
        enc = CKKSEncoder(self.params)
        
        self.encoder = enc
        self.encryptor = CKKSEncryptor(self.params, self.pk)
        self.evaluator = CKKSEvaluator(self.params)
        self.decryptor = dec

    # Utils
    def get_params(self):
        return self.params

    def get_public_key_set(self):
        return {"pk": self.pk, "relin_key":self.relin_key, "conj_key": self.conj_key, "rot_keys": self.rot_keys}

    # define the initial random values of the new latent vector v_j
    def newQuestion(self, k):
        vcl = np.random.rand(k+1) 
        vcl[0] = 0. #item bias
        
        pl = self.encoder.encode(self.prepare(vcl), self.scaling_factor)
        v = self.encryptor.encrypt(pl)
        
        return v, vcl    
    
    def decrypt(self, v):
        pl = self.decryptor.decrypt(v)
        clear = self.encoder.decode(pl)
        return [x.real for x in clear]

    def compare(self, ct, ct2, twoct=0, columns=["CKKS", "Clear"]):
        if twoct == 1:
            print(pd.DataFrame([[x.real, xc.real] for x,xc in zip(self.decrypt(ct), self.decrypt(ct2))], columns=["CKKS", "Clear"]).T)

        elif twoct == 0:
            print(pd.DataFrame([[x, xc] for x,xc in zip(self.decrypt(ct), ct2)], columns=["CKKS", "Clear"]).T)
        
        else:
            print(pd.DataFrame([[x, xc] for x,xc in zip(ct, ct2)], columns=columns))

