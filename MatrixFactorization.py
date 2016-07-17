
# coding: utf-8

# In[ ]:

#from IPython.core.display import display
import pandas as pd
import json
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from multiprocessing import Process, Pool
import multiprocessing
from scipy.linalg import solve

import time
import timeit

import numpy as np
import numba

import ctypes
import datetime
from math import sqrt
import platform
platform.architecture()
# %load_ext line_profiler
import scipy
import os
# %load_ext Cython
now = lambda: datetime.datetime.now()
# %load_ext line_profiler

# %matplotlib inline


# # Nacteni dat

# In[ ]:

directory = 'sample-data-v2/'
ndataset = 'goout'#"movielens_1m"
data_path = '/home/kuba/ownCloud/Recombee/'


with open(data_path+directory+ndataset+"/items.json",'r') as f:
    items = json.loads(f.read())

with open(data_path+directory+ndataset+"/properties.json",'r') as f:
    properties = json.loads(f.read())

with open(data_path+directory+ndataset+"/user.folds.json",'r') as f:
    user_folds = json.loads(f.read())

with open(data_path+directory+ndataset+"/users.int2str.json",'r') as f:
    users_int2str = json.loads(f.read())

with open(data_path+directory+ndataset+"/users.str2int.json",'r') as f:
    users_str2int = json.loads(f.read())

dataset = pd.read_csv(data_path+directory+ndataset+"/ratings.csv",  ) #dtype = {'rating': np.float, 'itemId': str, 'userId':str}
# ratings = SFrame.read_csv("/home/kuba/ownCloud/ModGen-fac-mat/sample-data-v2/"+ndataset+"/ratings.csv", column_type_hints=[str,str,float] )
with open(data_path+directory+ndataset+"/items.int2str.json",'r') as f:
    items_int2str = json.loads(f.read())

with open(data_path+directory+ndataset+"/items.str2int.json",'r') as f:
    items_str2int = json.loads(f.read())



def split_dataset(dataset, test_size, relevant):        
    dataset["Testset"] = False
    if(test_size == 0):
        return dataset[dataset.Testset==False], dataset[dataset.Testset==True] 
    
    relevant_ = dataset.loc[dataset['rating']>=relevant]
    test_indices = []
    for key, user_relevant in relevant_.groupby('userId'):
        if(user_relevant.shape[0]>=2):
            indeces = user_relevant.index.tolist()
            test_indices.extend(np.random.choice(indeces, int(np.ceil(len(indeces)*test_size)), replace = False))


    dataset.loc[test_indices, "Testset"] = True
    print("USER DONE")
    testset = dataset[dataset.Testset==True]
    trainset = dataset[dataset.Testset==False]

    len_m  = 0
    for key, trainset_ratings in dataset.groupby('itemId'):
        if (np.all(trainset_ratings['Testset'])):

            indeces = trainset_ratings.index.tolist()
            rand = np.random.choice(indeces, int(np.ceil(len(indeces)*0.1)), replace = False)
            len_m +=len(rand)
            dataset.loc[rand, 'Testset'] = False

    print("Move from test set to train set: ", len_m)

    return dataset[dataset.Testset==False], dataset[dataset.Testset==True]


# # Faktorizace

# In[ ]:

class MatrixFactorization:
    def __init__(self, trainset, testset, no_fold, relevant = .25, test_size = 0.2, ndataset= None,  users_str2int= None, items_str2int=None):
        self.users_str2int, self.items_str2int = users_str2int, items_str2int

        #hodnota, kdy je rating oznacen za relevantni
        self.relevant = relevant
        #velikost trenovaci mnoziny <0,1>
        self.test_size = test_size

        #slozka vyrazenych uzivatelu
        self.no_fold = no_fold


        #v init() prirazena matice latentnich vektoru
        self.Users = None
        self.Items = None

        #nazev datasetu, cislo testovaci slozky
        self.ndataset, self.no_fold = ndataset, no_fold

        self.Trainset = trainset
        self.Testset = testset

        self.no_ratings = self.Trainset.shape[0]

        #mnozina users
        self.Users_set = set([str(id_) for id_ in trainset['userId']])
        #mnozina items
        self.Items_set = set([str(id_) for id_ in trainset['itemId']])

        #pocet users
        self.no_Users = len(self.Users_set)
        #pocet items
        self.no_Items = len(self.Items_set)

        #mapovaci dictonary, pro snadnejsi praci s poli user_string : uniqe number
        self.user_map_dict = dict(zip(self.Users_set,range(self.no_Users)))
        self.item_map_dict = dict(zip(self.Items_set,range(self.no_Items)))

        #mapovaci lambda funkce list of users : list of mapped numbers
        self.u_map = lambda users : list(map(lambda id_: self.user_map_dict[id_], users))
        self.i_map = lambda items : list(map(lambda id_: self.item_map_dict[id_], items))

        self.Trainset['userId_'] = self.Trainset['userId'].apply(lambda id_: self.user_map_dict[str(id_)])
        self.Trainset['itemId_'] = self.Trainset['itemId'].apply(lambda id_: self.item_map_dict[str(id_)])


        self.Testset['userId_'] = self.Testset['userId'].apply(lambda id_: self.user_map_dict[str(id_)])
        self.Testset['itemId_'] = self.Testset['itemId'].apply(lambda id_: self.item_map_dict[str(id_)])

#         self.Item_pop = self.item_pop(self.Trainset)


        #TRAINSET se inicializuje az pri spusteni optimalizace



        #TESTSET
        #dictonary: userId : {"ids":[...], "ratings":[...], 'weights':[...]}
        #ids : indexy items
        self.User_Items_testset = self.columns_to_dict(self.Testset, 'userId_', ['itemId_','rating'])
        #dictonary: itemId : {"ids":[...], "ratings":[...], 'weights':[...]}
        #ids : indexy users
        self.Item_Users_testset = self.columns_to_dict(self.Testset, 'itemId_', ['userId_','rating'])

        #koeficient <0,1> ovlivnujici tendenci modelu doporucovat popularni items (1)
        self.beta = None


    def columns_to_dict(self, df, key_column, value_column):
        """Transformace pandas.DataFrame do dictonary"""
        dict_ = {}
        matrix = df[[key_column] + value_column].as_matrix()

        for row in matrix:
            key = int(row[0])
            ids = int(row[1])
            rating = row[2]
            if(matrix.shape[1]>3):
                weight = row[3]
            try:
                dict_[key]['ids'].append(ids)
                dict_[key]['ratings'].append(rating)
                if(matrix.shape[1]>3):
                    dict_[key]['weights'].append(weight)

            except:
                dict_[key] = {'ids':[], 'ratings':[], 'weights':[]}
                dict_[key]['ids'].append(ids)
                dict_[key]['ratings'].append(rating)
                if(matrix.shape[1]>3):
                    dict_[key]['weights'].append(weight)


        return dict_

    def item_pop(self, dataset):
        """Priradi kazdemu itemu pocet hodnoceni oznacenych jako relevantni (popularita itemu)
        dataset : pandas.DataFrame(({userId: string, itemId: string, itemId_ : number, rating: float}))
        ----------------------
        return dict({item_ : # relevant ratings})
        """
        dict_ = {}

        for itemId, dataframe in dataset.groupby('itemId_'):
            r_sum = np.sum(dataframe['rating']>=self.relevant)
            dict_[itemId] = r_sum

        return dict_

    def item_weight(self,dataset):
        dict_ = {}
        for itemId, dataframe in dataset.groupby('itemId_'):
            no_relevant_ratings = np.sum(dataframe['rating']>=self.relevant)
            dict_[itemId] = pow((1/(no_relevant_ratings+1)),self.beta)

        return dict_
    '''
    INIT
    '''
    def init(self):
        """ Inicializace matice latentnich vektoru users a items. Inicializace matice U.T*U a V.T*V"""
        if(self.random_init):#Chci nahodne inicializovat latentni vektory pri zapoceti optimalizace s jinymi parametry?
            if(self.multiprocessing):#Chci pouzit multiprocessing?
                #print("START INIT USER AND ITEMS FEATURE VECTORES")
                #Nehrozi paralelni pristup ke sdilenym zdrojum(radkum matice), neni potreba zamykat. Kazdy proces ma urcenou mnozinu radku, ktere updatuje.
                user_shared_array = np.frombuffer(multiprocessing.Array(ctypes.c_double, np.random.rand(self.no_Users * self.no_factors), lock=False),dtype=float)
                #Matice latentnich vektoru
                self.Users = user_shared_array.reshape(self.no_Users, self.no_factors)

                item_shared_array = np.frombuffer(multiprocessing.Array(ctypes.c_double, np.random.rand(self.no_Items * self.no_factors), lock= False),dtype=float)
                self.Items = item_shared_array.reshape(self.no_Items, self.no_factors)

                #Matice sdilena vsemi procesory U.T * U
                self.UU = np.frombuffer(multiprocessing.Array(ctypes.c_double, np.zeros(self.no_factors * self.no_factors), lock= False),dtype=float).reshape(self.no_factors, self.no_factors)
                self.VV = np.frombuffer(multiprocessing.Array(ctypes.c_double, np.zeros(self.no_factors * self.no_factors), lock= False),dtype=float).reshape(self.no_factors, self.no_factors)
                #print("FINISH INIT USER AND ITEMS FEATURE VECTORES")
            else:
                self.Users = np.random.rand(self.no_Users, self.no_factors)
                self.Items = np.random.rand(self.no_Items, self.no_factors)

                self.UU = np.random.rand(self.no_factors, self.no_factors)
                self.VV = np.random.rand(self.no_factors, self.no_factors)


    def init_optimizer(self, beta):
        """Kontrola nastavenych parametru a prirazeni vah"""

        if(self.weights_mode == "AllRank"):
            assert beta == 0, "Beta must be 0! Use AllRank-pop"

            if(self.beta is None or self.beta != beta):
                self.beta = 0
                self.Item_weight = self.item_weight(self.Trainset)
                self.init_users_items_dictonary()
            print("** Set weight: ",self.weight, " to missing ratings and ", set(self.Trainset['weight'])," to observate ratings **")

        elif(self.weights_mode == "AllRank-pop"):
            print("AllRank-pop")
            if(self.beta is None or self.beta != beta):
                self.beta = beta
                self.Item_weight = self.item_weight(self.Trainset)
                self.init_users_items_dictonary()
            print("** Set weight: ",self.weight, " to missing ratings", ", beta: ",self.beta," and avg weight ", self.Trainset['weight'].mean(), "**")

        elif(self.weights_mode == "MF-RMSE"):
            assert self.weight == 0, "Weight of missiong values MF-RMSE mode must be 0! Use AllRank or AllRank-pop"
            assert beta == 0, "Beta must be 0!"

            if(self.beta is None or self.beta != beta):
                self.beta = 0
                self.Item_weight = self.item_weight(self.Trainset)
                self.init_users_items_dictonary()
            print("** Set weight: ",self.weight, " to missing ratings **")

        if(self.imputation_value != 0 ):
            print("** Surrogate missing rating values by imputation value: ", self.imputation_value, " **")

    def init_users_items_dictonary(self):
        self.Trainset['weight'] = self.Trainset['itemId_'].apply(lambda id_: self.Item_weight[id_])
        weight_sum = self.Trainset['weight'].sum()
        self.Trainset['weight'] = self.Trainset['weight'].apply(lambda w: w/weight_sum*self.no_ratings)

        #dictonary: userId : {"ids":[...], "ratings":[...], 'weights':[...]} ; ids : indexy items
        self.User_Items = self.columns_to_dict(self.Trainset, 'userId_', ['itemId_','rating', 'weight'])
        #dictonary: itemId : {"ids":[...], "ratings":[...], 'weights':[...]} ; ids : indexy users
        self.Item_Users = self.columns_to_dict(self.Trainset, 'itemId_', ['userId_','rating', 'weight'])


    '''
    ATOP
    '''

    def ATOP(self, User_Items):
        """Vypocet ATOP http://users.cs.fiu.edu/~lzhen001/activities/KDD_USB_key_2010/docs/p713.pdf"""
#         d = now()
        pool = multiprocessing.Pool(processes = self.no_processes)
        nranks = pool.map_async(NRANKs_u, [(self.Users[user], self.User_Items[user]['ids'], self.Items) for user in self.User_Items.keys()])

        pool.close()
        pool.join()
        ATOP = np.mean([item for sublist in nranks.get() for item in sublist])
#         print("TIME ATOP ", now() - d)
        return ATOP


    '''
    RMSE
    '''
    def RMSE(self, Item_Users):
        """Vypocet RMSE"""
#         d = now()
        U = self.Users
        V = self.Items
        errors = []
        for item in Item_Users.keys():
            ratings = Item_Users[item]['ratings']
            users = Item_Users[item]['ids']

            users_latent = U[users]
            item_latent = V[item].T
            errors.extend((np.array(ratings) - (self.imputation_value + np.dot(users_latent, item_latent)))**2)

        rmse = sqrt(np.mean(errors))
#         print("RMSE time ", now() - d)
        return rmse


    '''
    LATENT FACTORS
    '''

    def items_factor(self,batch):
        """Update latentnich vektoru. Kazdy procesor dostane disjunktni "varku" latentnich vektoru ke zpracovani.
        batch: range(i,i+N)

        http://users.cs.fiu.edu/~lzhen001/activities/KDD_USB_key_2010/docs/p713.pdf
        """
        V = self.Items
        U = self.Users
        UU = self.UU
        lambda_, r_m  = self.lambda_, self.imputation_value,
        
        weight, no_factors,no_Items = self.weight, self.no_factors, self.no_Items
        eye = np.eye(no_factors)
        d = now()
        for i in batch:
            i_rated = self.Item_Users[i]['ids']
            U_s = U[i_rated,:]
            Wi = np.array([self.Item_Users[i]['weights']])

            lM = (np.asmatrix(self.Item_Users[i]['ratings']) - r_m).dot(np.multiply(Wi.T.dot(np.ones((1,no_factors))), U_s))
            rM = UU - (weight*U_s.T).dot(U_s) + np.multiply(U_s.T,  np.ones((no_factors,1)).dot(Wi)).dot(U_s)

            reg = lambda_ * (weight * (no_Items-len(Wi)) + (Wi-weight).sum()) * eye
            res = np.linalg.solve(rM+reg,lM.T)
            #Update latentniho vektoru items matice
            V[i,:] = res.flatten()
        print("ITEM TIME ", now() - d)


    def users_factor(self, batch):
        VV = self.VV
        U = self.Users
        V = self.Items
        lambda_, r_m = self.lambda_, self.imputation_value
        
        weight, no_factors,no_Users = self.weight, self.no_factors, self.no_Users
        eye = np.eye(no_factors)
        d = now()
        for u in batch:
            u_rated = self.User_Items[u]['ids']
            V_s = V[u_rated,:]
            Wu = np.array([self.User_Items[u]['weights']])

            lM = (np.asmatrix(self.User_Items[u]['ratings']) - r_m).dot(np.multiply(Wu.T.dot(np.ones((1,no_factors))), V_s))
            rM = VV - (weight*V_s.T).dot(V_s) + np.multiply(V_s.T, np.ones((no_factors,1)).dot(Wu)).dot(V_s)
            reg = lambda_ * (weight * (no_Users-Wu.shape[0]) + (Wu-weight).sum()) * eye
            res = np.linalg.solve(rM+reg,lM.T)
            #Update latentniho vektoru users matice
            U[u,:] = res.flatten()
        print("USER TIME ", now() - d)
    '''
    OPTIMIZE RMSE
    '''
    def optimize_rmse(self, beta):
        #inicializuj user/item matici latentnich vektoru a U.T * U, V.T * V (V: matice latentnich vektoru items)
        self.init()
        self.init_optimizer(beta)

        weighted_errors = []
        ATOPs = []

        step_item = int(np.ceil(len(self.Items)/float(self.no_processes)))
        step_user = int(np.ceil(len(self.Users)/float(self.no_processes)))

        #Rozdel do disjuktnich, stejne velikych varek
        item_range = [range(i,min(i+step_item, self.no_Items)) for i in range(0, self.no_Items, step_item)]
        user_range = [range(u,min(u+step_user, self.no_Users)) for u in range(0, self.no_Users, step_user)]

        print("*******************************")
        print("Lambda: ", self.lambda_)
        print("Impute value: ", self.imputation_value)
        print("Weight: ", self.weight)
        print("Beta: ", self.beta)
        print("Factors: ", self.no_factors)
        print("Number of processes: ", self.no_processes)
        print("Number of iterations: ", self.no_iterations)
        print("*******************************")

#         self.Testset = None
#         self.Trainset = None

        for ii in range(self.no_iterations):

            if(self.multiprocessing):
                #ITEMS latent vectors
                process = []
                
                d = now()
                self.UU[:] = (self.weight*self.Users.T).dot(self.Users)
                print("UU ", now() - d)
                
                d = now()
                for batch in item_range:
                    p = Process(target = self.items_factor, args = (batch,))
                    p.daemon = True
                    process.append(p)
                    p.start()

                [p.join() for p in process]
                print("Item time",  now()-d)

                #USERS latent vectors
                process = []
                
                d = now()
                self.VV[:] = (self.weight*self.Items.T).dot(self.Items)
                print("VV ", now() - d)
                
                d = now()
                for batch in user_range:
                    p = Process(target = self.users_factor, args = (batch,))
                    p.daemon = True
                    process.append(p)
                    p.start()

                [p.join() for p in process]
                print("User time ",  now()-d)

            else:
                print("Single process")
                self.UU[:] = (self.weight*self.Users.T).dot(self.Users)
                for batch in item_range:
                    self.items_factor(batch,)
                self.VV[:] = (self.weight*self.Items.T).dot(self.Items)
                for batch in user_range:
                    self.users_factor(batch,)


            print(ii,end=";")
            #vypocti RMSE na training set
#                 weighted_errors.append(self.RMSE(self.Item_Users))
            #vypocti RMSE na testset
            if((self.no_iterations-1) == ii):
                d = now()
                rmse = self.RMSE(self.Item_Users)
                #vypocti ATOP
                if(self.compute_ATOP):
                    atop = self.ATOP(self.Item_Users)
                    ATOPs.append(atop)
                    print("\nATOP ", atop, " RMSE ", rmse, " TIME ", now()-d)
                else:
                    ATOPs = [0]
                    print("\nRMSE", rmse, " TIME ", now()-d)

        return weighted_errors, ATOPs


    def optimaze(self, no_iterations=1, loss_function="RMSE",  lambda_ = 0.001,  no_processes = 1, no_factors = 50, random_init = True,
                 weights_mode = "MF-RMSE", weight = 0, imputation_value = 0, beta = 0, mlprocessing= False, save_matrix = False, compute_ATOP = False):

        if(not mlprocessing):
            self.no_processes = 1
        else:
            self.no_processes = no_processes

        self.multiprocessing = mlprocessing
        self.loss_function = loss_function
        self.lambda_ = lambda_
        self.no_factors = no_factors
        self.no_iterations = no_iterations
        self.random_init = random_init
        self.weights_mode = weights_mode
        self.weight = weight
        self.imputation_value = imputation_value
        self.compute_ATOP = compute_ATOP


        weighted_errors, ATOPs = self.optimize_rmse(beta)
        self.ATOPv = np.max(ATOPs)
        if(save_matrix):
            self.save_matrices()

#         self.plot_rmse(weighted_errors)
        return np.max(ATOPs)
    '''
    PLOT AND EXPLORE
    '''
    def plot_rmse(self, weighted_errors):
        plt.plot(np.log(weighted_errors), label="weighted error: "+str(weighted_errors[-1]))
        plt.ylabel("RMSE log scale")
        plt.xlabel("no iterations")

        plt.legend()
        plt.show()

    def explore(self):
        print("Explore trainset")
        relevant = self.relevant
        no_ratings = self.Trainset.shape[0]
        no_missing = self.no_Users * self.no_Items - no_ratings
        no_all = no_ratings + no_missing

        sizes = [no_ratings, no_missing]
        labels = ["ratings", "missings"]
        colors = ['yellowgreen', 'lightskyblue']

        #pozorovane vs. chybejici hodnoceni
        plt.pie(sizes, labels=labels,
                autopct='%1.1f%%', shadow=True, startangle=90, colors= colors)
        plt.axis('equal')
        plt.show()
        print("#{} rating, #{} missing".format(no_ratings, no_missing))
        print("Estimate offset w(m): ", no_ratings/no_missing)
        ratings = self.Trainset["rating"].values
        print("Avg of ratings: ", ratings.mean())

        #relevantni vs. irelevantni hodnoceni
        no_relevant = np.sum(ratings>=relevant)
        no_irelevant = np.sum(ratings<relevant)
        labels = ["relevant", "irrelevenat"]

        sizes = [no_relevant, no_irelevant]
        colors = ['lightskyblue', 'lightcoral']
        plt.pie(sizes, labels=labels,
                autopct='%1.1f%%', shadow=True, startangle=90, colors= colors)
        plt.axis('equal')
        plt.show()

        print("#{} relevant, #{} irelevant".format(no_relevant, no_irelevant))

        #rozlozeni popularity mezi items
        items_popularity = np.sort(list(self.Item_pop.values()))[::-1]
#         plt.bar(range(len(items_popularity)),items_popularity)
        plt.xticks([])
        plt.ylabel("# of item rating marked as relevant")
        plt.xlabel("item")
        plt.plot(items_popularity,'_')
        plt.show()

        #Histogram hodnoceni
        plt.hist(self.Trainset['rating'].values,bins = len(set(self.Trainset['rating'])))
        plt.xlabel("rating")
        plt.show()

    def save_matrices(self):
        d = now()
        no_iterations, no_factors, lambda_ , weight, r_m = self.no_iterations, self.no_factors, self.lambda_, self.weight, self.imputation_value
        ndataset, fold = self.ndataset, self.no_fold
        if not os.path.exists(data_path+"MATRICES/"+ndataset):
            os.makedirs(data_path+"MATRICES/"+ndataset)
        if(self.compute_ATOP):
            atop_suffix = "_ATOP:"+str(self.ATOPv)
        else:
            atop_suffix =""
            
        with open(data_path+"MATRICES/"+ndataset+"/"+ndataset+str(no_fold)+"_model_f:"+str(no_factors)+"l:"+str(lambda_)+"w:"+str(weight)+"b:"+str(self.beta)+"r:"+str(self.imputation_value)+atop_suffix+".txt", 'w') as f:
            f.write("m "+str(self.Users.shape[0])+"\n")
            f.write("n "+str(self.Items.shape[0])+"\n")
            f.write("k "+str(no_factors)+"\n")

            user_map = {v: k for k, v in self.user_map_dict.items()}
            for idx, laten_vec in enumerate(self.Users):
                string_idx = user_map[idx]
                idf = self.users_str2int[string_idx]
                f.write("p"+str(idf)+" "+' '.join(list(laten_vec.astype('str')))+"\n")

            item_map = {v: k for k, v in self.item_map_dict.items()}
            for idx, laten_vec in enumerate(self.Items):
                string_idx = item_map[idx]
                idf = self.items_str2int[string_idx]
                f.write("q"+str(idf)+" "+' '.join(list(laten_vec.astype('str')))+"\n")

        #print("************ UKLADANI MATICE ", now()-d)

def NRANKs_u(args):
    """Vypocti normalizovany rank pro uzivatele.
    args: user latentni vektor, user testovaci items, matice latentnich vektoru items
    """
    user_vec, items, V = args

    ratings_hat = np.dot(user_vec, V.T)
    N = ratings_hat.shape[0]
    ratings = ratings_hat[items]
    nranks = np.array(list(map(lambda rating: np.sum(ratings_hat<rating), ratings)))/N

    return nranks


# # MAIN

# In[ ]:

if __name__ == "__main__":
    test_size, relevant = 0.0, 1
    SAVE_MATRIX = True
    ATOP = False
    #[0,1,2,3,4,5,6,7,8,9]
    for no_fold in [0,1,2,3,4,5,6,7,8,9]:  #iteruj pres testovaci slozky
        #odstran z datasetu testovaci users obsazene ve slozce
        ratings = dataset[~dataset.userId.isin(user_folds[no_fold])]
        trainset, testset = split_dataset(ratings.copy(),test_size = test_size, relevant = relevant)

        MFact = MatrixFactorization(trainset, testset, no_fold = no_fold , test_size = test_size, relevant = relevant, ndataset = ndataset,
                                    users_str2int= users_str2int, items_str2int = items_str2int)
        #[0, 0.0008, 0.001, 0.0015 ,0.002, 0.005, 0.01, 0.02]
        for lambda_ in [0, 0.001, 0.005, 0.008, 0.01]: #iteruj pres lambda
            #[1, 2, 5, 10, 30, 50, 100,200, 300],
            for no_factor in [30,100]: # iteruj pres delku latentnich vektoru
                for no_iterations in [6]: #iteruj pres pocet iteraci alternating least square
                    #[-2,-1,-0.5,-0.2,0, 0.2, 0.5, 1, 2]
                    for beta in [0, 0.1, 0.2]:
                        for weight in [0.02, 0.05, 0.08, 0.1]:
                            for imputation_value in [0, 0.01]:
                                for p in [5]:
                                    d = now()
                                    ATOP = MFact.optimaze(no_iterations = no_iterations,  lambda_ = lambda_, no_processes = p, no_factors = no_factor,
                                                          beta = beta,
                                                          weights_mode = "AllRank-pop", weight = weight, imputation_value = imputation_value,
                                                          random_init = True, mlprocessing = True, save_matrix = SAVE_MATRIX, compute_ATOP = ATOP)

                                    print("********** TIME ",p, now() - d)
                                    print("************************************")


# # Line profiler

# In[ ]:

# %load_ext line_profiler
# %lprun -f MFact.users_factor MFact.users_factor(range(0,1000))


# In[ ]:

ITEM TIME  0:00:03.226633
ITEM TIME  0:00:03.417261
ITEM TIME  0:00:03.701171
ITEM TIME  0:00:03.598445
ITEM TIME  0:00:03.758560

