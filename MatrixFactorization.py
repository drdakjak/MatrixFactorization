import numpy as np
import multiprocessing

class MatrixFactorization:

    def __init__(self, trainset, testset, user, product, rating, relevant=None):
        import datetime
        self.user_id = user
        self.product_id = product
        self.rating_id = rating
        
        self.now = lambda: datetime.datetime.now()
        self.rating = rating
        self.relevant = relevant

        self.Users = None
        self.Items = None

        self.Trainset = trainset.copy()
        self.Testset = testset.copy()

        self.no_ratings = self.Trainset.shape[0]

        self.Users_set = set([str(id_) for id_ in trainset[user]])
        self.Items_set = set([str(id_) for id_ in trainset[product]])
        
        self.no_Users = len(self.Users_set)
        self.no_Items = len(self.Items_set)

        #mapovaci dictonary, pro snadnejsi praci s poli user_string : uniqe number
        self.user_map_dict = dict(zip(self.Users_set,range(self.no_Users)))
        self.item_map_dict = dict(zip(self.Items_set,range(self.no_Items)))

        #mapovaci lambda funkce list of users : list of mapped numbers
        self.u_map = lambda users: list(map(lambda id_: self.user_map_dict[id_], users))
        self.i_map = lambda items: list(map(lambda id_: self.item_map_dict[id_], items))

        self.Trainset['user_'] = self.Trainset[user].apply(lambda id_: self.user_map_dict[str(id_)])
        self.Trainset['product_'] = self.Trainset[product].apply(lambda id_: self.item_map_dict[str(id_)])

        self.Testset['user_'] = self.Testset[user].apply(lambda id_: self.user_map_dict[str(id_)])
        self.Testset['product_'] = self.Testset[product].apply(lambda id_: self.item_map_dict[str(id_)])


        #TESTSET
        self.User_Items_testset = self.columns_to_dict(self.Testset, key='user_', keys_val=['product_',rating])
        self.Item_Users_testset = self.columns_to_dict(self.Testset, key='product_', keys_val=['user_',rating])

        #koeficient <0,1> ovlivnujici tendenci modelu doporucovat popularni items (1)
        self.beta = None
        


    def columns_to_dict(self, df, key, keys_val):
        """Transformace pandas.DataFrame do dictonary"""
        dict_ = {}
        df_ = df[[key] + keys_val]
        keys_ = ['ids', 'ratings', 'weights']
        for key, group in df_.groupby(key):
            dict_[key] = {}
            for idx, key_val in enumerate(keys_val):
                dict_[key][keys_[idx]] = group[key_val].tolist()

        return dict_

    def item_pop(self, dataset):
        """Priradi kazdemu itemu pocet hodnoceni oznacenych jako relevantni (popularita itemu)
        dataset : pandas.DataFrame(({user: string, product: string, product_ : number, rating: float}))
        ----------------------
        return dict({item_ : # relevant ratings})
        """
        dict_ = {}
        for product, dataframe in dataset.groupby('product_'):
            dict_[product] = np.sum(dataframe[self.rating]>=self.relevant)

        return dict_

    def item_weight(self,dataset):
        dict_ = {}
        for product, dataframe in dataset.groupby('product_'):
            no_relevant_ratings = np.sum(dataframe[self.rating]>=self.relevant)
            dict_[product] = pow((1/(no_relevant_ratings+1)),self.beta)

        return dict_
    '''
    INIT
    '''
    def init(self):
        from sys import getsizeof
        import ctypes

        """ Inicializace matice latentnich vektoru users a items. Inicializace matice U.T*U a V.T*V"""
        if(self.random_init):
            if(self.no_processes>1):
                
                self.max_no_Users = int(self.no_Users*1.5)
                self.user_shared_array = multiprocessing.Array(ctypes.c_float, np.random.rand(self.max_no_Users * self.no_factors), lock=False)
                self.Users = np.frombuffer(self.user_shared_array,dtype=np.float32)[:self.no_Users * self.no_factors].reshape(self.no_Users, self.no_factors)
                print("USER SIZE FLOAT", self.Users.nbytes/1e6, "MB")

                self.max_no_Items = int(self.no_Items*1.5)
                self.item_shared_array = multiprocessing.Array(ctypes.c_float, np.random.rand(self.max_no_Items * self.no_factors)
                self.Items = np.frombuffer(self.item_shared_array[:self.no_Items * self.no_factors], lock= False),dtype=np.float32.reshape(self.no_Items, self.no_factors)
                print("ITEM SIZE FLOAT", self.Items.nbytes/1e6, "MB")

#                 self.UU = np.frombuffer(multiprocessing.Array(ctypes.c_float, np.random.rand(self.no_factors * self.no_factors), lock= False),dtype=np.float32).reshape(self.no_factors, self.no_factors)
#                 print("UU SIZE FLOAT", self.UU.nbytes/1e6, "MB")
#                 self.VV = np.frombuffer(multiprocessing.Array(ctypes.c_float, np.random.rand(self.no_factors * self.no_factors), lock= False),dtype=np.float32).reshape(self.no_factors, self.no_factors)
#                 print("VV SIZE FLOAT", self.VV.nbytes/1e6, "MB")


            else:
                print("INIT FACTOR MATRICES")
                print(self.no_Users, self.no_factors,self.no_Items, self.no_factors)
                self.Users = np.random.rand(self.no_Users, self.no_factors)
                self.Items = np.random.rand(self.no_Items, self.no_factors)

        self.UU = np.random.rand(self.no_factors, self.no_factors)
        self.VV = np.random.rand(self.no_factors, self.no_factors)


    def init_optimizer(self):
        """Kontrola nastavenych parametru a prirazeni vah"""
        beta = self.beta
        if(self.weights_mode == "MF-RMSE"):
            assert self.weight == 0, "Weight of missiong values MF-RMSE mode must be 0! Use AllRank or AllRank-pop"
            assert beta == 0, "Beta must be 0!"
            self.init_users_items_dictonary()
            print("** Set weight: ",self.weight, " to missing ratings **")
            
        elif(self.weights_mode == "AllRank"):
            assert beta == 0, "Beta must be 0! Use AllRank-pop"
            self.init_users_items_dictonary()
            print("** Set weight: ",self.weight, " to missing ratings **")

        elif(self.weights_mode == "AllRank-pop"):
            print("AllRank-pop")
            if(self.beta is None or self.beta != beta):
                self.beta = beta
                self.Item_weight = self.item_weight(self.Trainset)
                self.init_users_items_dictonary()
            print("** Set weight: ",self.weight, " to missing ratings", ", beta: ",self.beta," and avg weight ", self.Trainset['weight'].mean(), "**")

        if(self.imputation_value != 0 ):
            print("** Surrogate missing rating values by imputation value: ", self.imputation_value, " **")

    def init_users_items_dictonary(self):
        if(self.weights_mode == "AllRank-pop"):
            self.Trainset['weight'] = self.Trainset['product_'].apply(lambda id_: self.Item_weight[id_])
            weight_sum = self.Trainset['weight'].sum()
            self.Trainset['weight'] = self.Trainset['weight'].apply(lambda w: w/weight_sum*self.no_ratings)
        else:
            self.Trainset['weight'] = 1
        
        self.User_Items = self.columns_to_dict(self.Trainset, 'user_', ['product_',self.rating, 'weight'])
        self.Item_Users = self.columns_to_dict(self.Trainset, 'product_', ['user_',self.rating, 'weight'])


    '''
    RMSE
    '''
    def RMSE(self, Item_Users):
        """Vypocet RMSE"""
#         d = self.now()
        U = self.Users
        V = self.Items
        errors = []
        for item in Item_Users.keys():
            ratings = Item_Users[item]['ratings']
            users = Item_Users[item]['ids']

            users_latent = U[users]
            item_latent = V[item].T
            errors.extend((np.array(ratings) - (self.imputation_value + np.dot(users_latent, item_latent)))**2)

        rmse = np.sqrt(np.mean(errors))
#         print("RMSE time ", self.now() - d)
        return rmse

    '''
    ATOP
    '''

    def ATOP(self, User_Items):
        """Vypocet ATOP https://github.com/drdakjak/MatrixFactorization/blob/master/p713.pdf"""
#         d = now()
        pool = multiprocessing.Pool(processes = self.no_processes)
        nranks = pool.map_async(NRANKs_u, [(self.Users[user], User_Items[user]['ids'], self.Items) for user in User_Items.keys()])

        pool.close()
        pool.join()
        ATOP = np.mean([item for sublist in nranks.get() for item in sublist])
#         print("TIME ATOP ", now() - d)
        return ATOP

    '''
    ADD NEW RECORD
    '''
    
    def add_record(self, record, is_test_case=False):
        user_id, item_id, rating = record[self.user_id], record[self.item_id], record[self.rating_id]
        
        if user_id in self.user_map_dict:
            if is_test_case:
                self.User_Items_testset[user_id]['ids'].append(item_id)
                self.User_Items_testset[user_id]['ratings'].append(rating) 
                self.User_Items_testset[user_id]['weights'].append(weight) # TODO weight      
            else:
                self.User_Items[user_id]['ids'].append(item_id)
                self.User_Items[user_id]['ratings'].append(rating) 
                self.User_Items[user_id]['weights'].append(weight) # TODO weight                                     
        else:
            
            if self.no_Users+1 > self.max_no_Users:
                print("Reallocation user matrix")
                self.max_no_Users = int(self.max_no_Users*1.5)
                user_shared_array_ = multiprocessing.Array(ctypes.c_float,
                                                           np.append(
                                                               self.Users.reshape(self.no_Users * self.no_factors),
                                                               np.random.rand((self.max_no_Users - self.no_Users) * self.no_factors)
                                                                   ), lock=False)                                          
            
                del self.user_shared_array
                self.user_shared_array = user_shared_array_
                
            self.user_map_dict[user_id] = self.no_Users
            if is_test_case:
                assert False, "Any record in training set for user"
            else:
                self.User_Items[self.no_Users] = {'ids': [item_id], 'ratings': [rating], 'weights': []} # TODO weight

            self.no_Users += 1
            self.Users_set.add(user_id)
            self.Users = np.frombuffer(self.user_shared_array,dtype=np.float32)[:self.no_Users * self.no_factors].reshape(self.no_Users, self.no_factors)
            

                                                               
        if item_id in self.item_map_dict:
            if is_test_case:
                self.Item_Users_testset[item_id]['ids'].append(user_id)
                self.Item_Users_testset[item_id]['ratings'].append(rating) 
                self.Item_Users_testset[item_id]['weights'].append(weight) # TODO weight  
            else:
                self.Item_Users[item_id]['ids'].append(user_id)
                self.Item_Users[item_id]['ratings'].append(rating) 
                self.Item_Users[item_id]['weights'].append(weight) # TODO weight                                     
        else:
            if self.no_Items+1 > self.max_no_Items:
                print("Reallocation product matrix")
                self.max_no_Items = int(self.max_no_Items*1.5)
                item_shared_array_ = multiprocessing.Array(ctypes.c_float,
                                                           np.append(
                                                               self.Items.reshape(self.no_Items * self.no_factors),
                                                               np.random.rand((self.max_no_Items - self.no_Items) * self.no_factors)
                                                                   ), lock=False)                                          
            
                del self.item_shared_array
                self.item_shared_array = item_shared_array_
                
            self.item_map_dict[item_id] = self.no_Items
            if is_test_case:
                assert False, "Any record in training set for product"
            else:
                self.Item_Users[self.no_Items] = {'ids': [user_id], 'ratings': [rating], 'weights': []} # TODO weight

            self.no_Items += 1
            self.Items_set.add(item_id)
            self.Items = np.frombuffer(self.item_shared_array,dtype=np.float32)[:self.no_Items * self.no_factors].reshape(self.no_Items, self.no_factors)
           
        UU = self.UU[:]
        VV = self.VV[:]
        for _ in range(6):
            user_id_ = self.user_map_dict[user_id]                                                
            self.UU[:] = UU + (self.weight*self.Users[user_id_].T).dot(self.Users[user_id_])
            self.items_factor([self.item_map_dict[item_id]])
            
            item_id_ = self.user_map_dict[item_id]
            self.VV[:] = VV + (self.weight*self.Items[item_id_].T).dot(self.Items[[item_id_]])                                   
            self.users_factor([self.user_map_dict[user_id]])

    
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
        eye = np.eye(no_factors, dtype=np.float32)
        d = self.now()
        for i in batch:
            item_users = self.Item_Users[i]
            i_rated = item_users['ids']
            U_s = np.take(U, i_rated, axis=0)
            Wi = np.array([item_users['weights']], dtype = np.float32)

            lM = (np.array([item_users['ratings']], dtype = np.float32) - r_m).dot(np.multiply(Wi.T.dot(np.ones((1,no_factors))), U_s))
            rM = UU - (weight*U_s.T).dot(U_s) + np.multiply(U_s.T,  np.ones((no_factors,1)).dot(Wi)).dot(U_s)
            reg = lambda_ * (weight * (no_Items-len(Wi)) + (Wi-weight).sum()) * eye
            res = np.linalg.solve(rM+reg,lM.T)
            #Update latentniho vektoru items matice
            V[i,:] = res.ravel()
        print("ITEM TIME ", str(self.now() - d))


    def users_factor(self, batch):
        VV = self.VV
        U = self.Users
        V = self.Items
        lambda_, r_m = self.lambda_, self.imputation_value

        weight, no_factors,no_Users = self.weight, self.no_factors, self.no_Users
        eye = np.eye(no_factors, dtype = np.float32)
        d = self.now()
        for u in batch:
            user_items = self.User_Items[u]
            u_rated = user_items['ids']

            V_s = np.take(V, u_rated, axis=0)
            Wu = np.array([user_items['weights']], dtype = np.float32)

                
            lM = (np.array([user_items['ratings']], dtype = np.float32) - r_m).dot(np.multiply(Wu.T.dot(np.ones((1,no_factors))), V_s))
            rM = VV - (weight*V_s.T).dot(V_s) + np.multiply(V_s.T, np.ones((no_factors,1)).dot(Wu)).dot(V_s)
            reg = lambda_ * (weight * (no_Users-Wu.shape[0]) + (Wu-weight).sum()) * eye
            res = np.linalg.solve(rM+reg,lM.T)
            #Update latentniho vektoru users matice
            U[u,:] = res.ravel()
        print("USER TIME ", str(self.now() - d))

    '''
    OPTIMIZE RMSE
    '''
    def optimize_rmse(self):
        #inicializuj user/item matici latentnich vektoru a U.T * U, V.T * V (V: matice latentnich vektoru items)
        
        self.init()
        self.init_optimizer()
        
        weighted_errors_test = []
        weighted_errors_train = []

        ATOPs = []
        
        print(self.Items.shape[0], self.no_processes)
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

            if(self.no_processes>1):
                #ITEMS latent vectors
                process = []

                d = self.now()
                self.UU[:] = (self.weight*self.Users.T).dot(self.Users)
                print("UU ", self.now() - d)
                d = self.now()
                for batch in item_range:
                    p = multiprocessing.Process(target = self.items_factor, args = (batch,))
                    p.daemon = True
                    process.append(p)
                    p.start()

                [p.join() for p in process]
                print("Item time",  self.now()-d)
                #USERS latent vectors
                process = []

                d = self.now()
                self.VV[:] = (self.weight*self.Items.T).dot(self.Items)
                print("VV ", self.now() - d)

                d = self.now()
                for batch in user_range:
                    p = multiprocessing.Process(target = self.users_factor, args = (batch,))
                    p.daemon = True
                    process.append(p)
                    p.start()

                [p.join() for p in process]
                print("User time ",  self.now()-d)

            else:
                print("Single process")
                self.UU[:] = (self.weight*self.Users.T).dot(self.Users)
                for batch in item_range:
                    self.items_factor(batch,)
                self.VV[:] = (self.weight*self.Items.T).dot(self.Items)
                for batch in user_range:
                    self.users_factor(batch,)


            print(ii)
            if True: #((self.no_iterations-1) == ii):
                d = self.now()
                weighted_errors_test.append(self.RMSE(self.Item_Users_testset))
                weighted_errors_train.append(self.RMSE(self.Item_Users))

                print("Train RMSE", weighted_errors_train[-1], " TIME ", str(self.now()-d))
                print("Test RMSE", weighted_errors_test[-1], " TIME ", str(self.now()-d))
        
        ATOP = self.ATOP(self.User_Items_testset)
        return weighted_errors_test, ATOP


    def solve(self, no_iterations=1, loss_function="RMSE",  lambda_=0.001,  no_processes=1, no_factors=50, random_init=True,
                 weights_mode="MF-RMSE", weight=0, imputation_value=0, beta=0):

        self.no_factors, self.lambda_, self.loss_function = no_factors, lambda_, loss_function 
        self.no_iterations, self.no_processes, self.random_init = no_iterations, no_processes, random_init
        self.beta, self.imputation_value, self.weight, self.weights_mode = beta, imputation_value, weight, weights_mode

                                                               
        self.weighted_errors, self.ATOP = self.optimize_rmse()
        
        row_factors = []
        user_map = {v: k for k, v in self.user_map_dict.items()}
        for idx, laten_vec in enumerate(self.Users):
            string_idx = user_map[idx]
            row_factors.append((string_idx, laten_vec))
        df_row_factors = pd.DataFrame(data=row_factors, columns=['id','factors'])
        
        column_factors = []
        item_map = {v: k for k, v in self.item_map_dict.items()}
        for idx, laten_vec in enumerate(self.Items):
            string_idx = item_map[idx]
            column_factors.append((string_idx, laten_vec))
        df_column_factors = pd.DataFrame(data=column_factors, columns=['id','factors'])

        return df_row_factors, df_column_factors
    
    '''
    PLOT AND EXPLORE
    '''
    def plot_rmse(self):
        import matplotlib.pyplot as plt
        plt.style.use('ggplot')
        %matplotlib inline
        weighted_errors = self.weighted_errors
        plt.plot(np.log(weighted_errors), label="weighted error: "+str(weighted_errors[-1]))
        plt.ylabel("RMSE log scale")
        plt.xlabel("no iterations")

        plt.legend()
        plt.show()

    def explore(self):
        rating = self.rating
        import matplotlib.pyplot as plt
        plt.style.use('ggplot')
        %matplotlib inline

        print("Explore trainset")
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
        ratings = self.Trainset[rating].values
        print("Avg of ratings: ", ratings.mean())

#         #relevantni vs. irelevantni hodnoceni
#         no_relevant = np.sum(ratings>=relevant)
#         no_irelevant = np.sum(ratings<relevant)
#         labels = ["relevant", "irrelevenat"]

#         sizes = [no_relevant, no_irelevant]
#         colors = ['lightskyblue', 'lightcoral']
#         plt.pie(sizes, labels=labels,
#                 autopct='%1.1f%%', shadow=True, startangle=90, colors= colors)
#         plt.axis('equal')
#         plt.show()

#         print("#{} relevant, #{} irelevant".format(no_relevant, no_irelevant))

        #rozlozeni popularity mezi items
#         items_popularity = np.sort(list(self.Item_pop.values()))[::-1]
# #         plt.bar(range(len(items_popularity)),items_popularity)
#         plt.xticks([])
#         plt.ylabel("# of item rating marked as relevant")
#         plt.xlabel("item")
#         plt.plot(items_popularity,'_')
#         plt.show()

        #Histogram hodnoceni
        plt.hist(self.Trainset[rating].values,bins = len(set(self.Trainset[rating])))
        plt.xlabel("rating")
        plt.show()
        
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
