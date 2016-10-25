class MatrixFactorization:

    def __init__(self, trainset, testset, rowId, columnId, ratingId, relevant=None):
        import datetime
        import numpy as np

        
        self.now = lambda: datetime.datetime.now()
        self.ratingId = ratingId
        self.relevant = relevant

        self.Users = None
        self.Items = None

        self.Trainset = trainset.copy()
        self.Testset = testset.copy()

        self.no_ratings = self.Trainset.shape[0]

        self.Users_set = set([str(id_) for id_ in trainset[rowId]])
        self.Items_set = set([str(id_) for id_ in trainset[columnId]])
        
        self.no_Users = len(self.Users_set)
        self.no_Items = len(self.Items_set)

        #mapovaci dictonary, pro snadnejsi praci s poli user_string : uniqe number
        self.user_map_dict = dict(zip(self.Users_set,range(self.no_Users)))
        self.item_map_dict = dict(zip(self.Items_set,range(self.no_Items)))

        #mapovaci lambda funkce list of users : list of mapped numbers
        self.u_map = lambda users: list(map(lambda id_: self.user_map_dict[id_], users))
        self.i_map = lambda items: list(map(lambda id_: self.item_map_dict[id_], items))

        self.Trainset['rowId_'] = self.Trainset[rowId].apply(lambda id_: self.user_map_dict[str(id_)])
        self.Trainset['columnId_'] = self.Trainset[columnId].apply(lambda id_: self.item_map_dict[str(id_)])

        self.Testset['rowId_'] = self.Testset[rowId].apply(lambda id_: self.user_map_dict[str(id_)])
        self.Testset['columnId_'] = self.Testset[columnId].apply(lambda id_: self.item_map_dict[str(id_)])


        #TESTSET
        self.User_Items_testset = self.columns_to_dict(self.Testset, key='rowId_', keys_val=['columnId_',ratingId])
        self.Item_Users_testset = self.columns_to_dict(self.Testset, key='columnId_', keys_val=['rowId_',ratingId])

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
                dict_[key][keys_[idx]] = group[key_val].tolist() # TODO storage as np.array

        return dict_

    def item_pop(self, dataset):
        """Priradi kazdemu itemu pocet hodnoceni oznacenych jako relevantni (popularita itemu)
        dataset : pandas.DataFrame(({rowId: string, columnId: string, columnId_ : number, rating: float}))
        ----------------------
        return dict({item_ : # relevant ratings})
        """
        dict_ = {}
        for columnId, dataframe in dataset.groupby('columnId_'):
            dict_[columnId] = np.sum(dataframe[self.ratingId]>=self.relevant)

        return dict_

    def item_weight(self,dataset):
        dict_ = {}
        for columnId, dataframe in dataset.groupby('columnId_'):
            no_relevant_ratings = np.sum(dataframe[self.ratingId]>=self.relevant)
            dict_[columnId] = pow((1/(no_relevant_ratings+1)),self.beta)

        return dict_
    '''
    INIT
    '''
    def init(self):
        from sys import getsizeof
        """ Inicializace matice latentnich vektoru users a items. Inicializace matice U.T*U a V.T*V"""
        if(self.random_init):#Chci nahodne inicializovat latentni vektory pri zapoceti optimalizace s jinymi parametry?
            if(self.no_processes>1):#Chci pouzit multiprocessing?
                user_shared_array = np.frombuffer(multiprocessing.Array(ctypes.c_float, np.random.rand(self.no_Users * self.no_factors), lock=False),dtype=np.float32)
                self.Users = user_shared_array.reshape(self.no_Users, self.no_factors)
                print("USER SIZE FLOAT", self.Users.nbytes, "bytes")

                item_shared_array = np.frombuffer(multiprocessing.Array(ctypes.c_float, np.random.rand(self.no_Items * self.no_factors), lock= False),dtype=np.float32)
                self.Items = item_shared_array.reshape(self.no_Items, self.no_factors)
                print("ITEM SIZE FLOAT", self.Items.nbytes, "bytes")

                self.UU = np.frombuffer(multiprocessing.Array(ctypes.c_float, np.random.rand(self.no_factors * self.no_factors), lock= False),dtype=np.float32).reshape(self.no_factors, self.no_factors)
                print("UU SIZE FLOAT", self.UU.nbytes, "bytes")
                self.VV = np.frombuffer(multiprocessing.Array(ctypes.c_float, np.random.rand(self.no_factors * self.no_factors), lock= False),dtype=np.float32).reshape(self.no_factors, self.no_factors)
                print("VV SIZE FLOAT", self.VV.nbytes, "bytes")


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
            self.Trainset['weight'] = self.Trainset['columnId_'].apply(lambda id_: self.Item_weight[id_])
            weight_sum = self.Trainset['weight'].sum()
            self.Trainset['weight'] = self.Trainset['weight'].apply(lambda w: w/weight_sum*self.no_ratings)
        else:
            self.Trainset['weight'] = 1
        
        self.User_Items = self.columns_to_dict(self.Trainset, 'rowId_', ['columnId_',self.ratingId, 'weight'])
        self.Item_Users = self.columns_to_dict(self.Trainset, 'columnId_', ['rowId_',self.ratingId, 'weight'])


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
                print(self.Items[0])
                d = self.now()
                for batch in item_range:
                    p = Process(target = self.items_factor, args = (batch,))
                    p.daemon = True
                    process.append(p)
                    p.start()

                [p.join() for p in process]
                print("Item time",  self.now()-d)
                print(self.Items[0])
                #USERS latent vectors
                process = []

                d = self.now()
                self.VV[:] = (self.weight*self.Items.T).dot(self.Items)
                print("VV ", self.now() - d)

                d = self.now()
                for batch in user_range:
                    p = Process(target = self.users_factor, args = (batch,))
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

        return weighted_errors_test


    def solve(self, no_iterations=1, loss_function="RMSE",  lambda_=0.001,  no_processes=1, no_factors=50, random_init=True,
                 weights_mode="MF-RMSE", weight=0, imputation_value=0, beta=0):

        self.loss_function = loss_function
        self.lambda_ = lambda_
        self.no_factors = no_factors
        self.no_iterations = no_iterations
        self.no_processes = no_processes
        self.random_init = random_init
        self.weights_mode = weights_mode
        self.weight = weight
        self.imputation_value = imputation_value
        self.beta = beta

        self.weighted_errors = self.optimize_rmse()
        
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
