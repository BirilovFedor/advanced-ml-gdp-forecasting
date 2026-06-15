import numpy as np


class LSTM:

    def __init__(self, batch_size, number_of_features, hidden_size, lenght_of_output_window):

        '''
        Docstring for __init__

        :param c: long-term memory
        :param h: short-term memory

        :param wf: weights used in making a decision what percent of long-term memory to keep
        :param bf: bias used in making a decision what percent of long-term memory to keep
        :param wi: weights used in making a decision what percent of potential long-term memory to keep
        :param bi: bias used in making a decision what percent of potential long-term memory to keep
        :param wc: weights used in creating potential long-term memory
        :param bc: bias used in creating potential long-term memory
        :param wo: weights used in making a decision what percent of long-term memory to use in current short-term memory
        :param bo: bias used in making a decision what percent of long-term memory to use in current short-term memory
        '''
        

        self.c = np.zeros((batch_size, hidden_size))
        self.h = np.zeros((batch_size, hidden_size))

        self.wf = np.random.randn(hidden_size + number_of_features, hidden_size) * 0.1
        self.bf = np.zeros((1, hidden_size))
        self.wi = np.random.randn(hidden_size + number_of_features, hidden_size) * 0.1
        self.bi = np.zeros((1, hidden_size))
        self.wc = np.random.randn(hidden_size + number_of_features, hidden_size) * 0.1
        self.bc = np.zeros((1, hidden_size))
        self.wo = np.random.randn(hidden_size + number_of_features, hidden_size) * 0.1
        self.bo = np.zeros((1, hidden_size))
        self.wy = np.random.randn(hidden_size, lenght_of_output_window) * 0.1
        self.by = np.zeros((1, lenght_of_output_window))
        self.memory = []
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.number_of_features = number_of_features

    def sigmoid_function(self, x):

        '''
        Docstring for sigmoid_function

        function takes x-axis value and returns corresponding y-axis value on the sigmoid function
        the returned value y in range [0, 1], can be treated as probability

        :param x: x-axis values
        '''

        y = 1 / (1 + np.exp(-x))
        return y
    
    def tahn_function(self, x):

        '''
        Docstring for tahn_function

        function takes x-axis value and returns corresponding y-axis value on the tahn function
        the returned value y in range [-1, 1]

        :param x: x-axis values
        '''

        y = (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))
        return y

    def preactivation_function(self, x, w, b):

        '''
        Docstring for preactivation_function

        function that calculates the preactivation, linear transformation of x and w

        :param x: data + previous hidden layer(h or short-term memory)
        :param w: weights
        :param b: bias

        '''

        return np.dot(x, w) + b

    def memory_cleaner(self):
        '''
        Docstring for memory_cleaner

        function is used to clean the long-term and short term memories for new data
        '''
        self.memory = []
        self.c[:] = 0
        self.h[:] = 0

    def forward_sequence(self, x_sequence):
        '''
        Docstring for forward_sequence

        :param x_sequence: matrix of a size batch_size x number_of_input_years x number_of_features

        function runs forward pass for every year to update the long-term and short term memories
        '''

        self.memory_cleaner()


        for t in range (x_sequence.shape[1]):
            self.forward_pass(x_sequence[:, t, :])
        
    def forward_pass(self, x):

        ''' 
        Docstring for forward_pass

        :param x: parameter x is size of batch x number of features

        this function is taking 1 year of data and updating the long and short term memory of LSTM memory
        by using 3 steps: 
            1) decide what percent of long-term memory to keep
            2) decide how much to add to long-term memory
            3) decide how much of long-rem memory to reveal to short-term memory
        '''
        # save previous c values
        prev_c = self.c.copy()
        prev_h = self.h.copy()

        ################################## step 1: forget gate #########################################

        # linear combination of data and short-term memory
        f = self.preactivation_function(np.concatenate((self.h, x), axis=1), self.wf, self.bf)
        # use sigmoid to get percent of long-term memory that is kept from last iteration
        active_f = self.sigmoid_function(f)

        ################################## step 2: decide what amount of potential long-term meory to add ################

        # linear combination of data and short-term memory
        i = self.preactivation_function(np.concatenate((self.h, x), axis=1), self.wi, self.bi)
        # use sigmoid to get percent of potential long-term memory to add
        active_i = self.sigmoid_function(i)

        ################################## step 3: update gate #########################################

        # linear combination of data and short-term memory
        Ct = self.preactivation_function(np.concatenate((self.h, x), axis=1), self.wc, self.bc)
        # use tahn to get potential long-term memory to add
        active_ct = self.tahn_function(Ct)
        # keep part of old long-term memory and add new long-term memory
        self.c = self.c * active_f + active_ct * active_i

        ################################## step 4: output gate #########################################

        # linear combination of data and short-term memory
        o = self.preactivation_function(np.concatenate((self.h, x), axis=1), self.wo, self.bo)
        # use sigmoid to get percent of potential long-term memory to reveal to short-term memory in a current state
        active_o = self.sigmoid_function(o)

        self.h = active_o*self.tahn_function(self.c)

        self.memory.append({
            "o": active_o,
            "i": active_i,
            "ct": active_ct,
            "f": active_f,
            "prev_c": prev_c,
            "c": self.c.copy(),
            "prev_h": prev_h,
            "combined": np.concatenate((prev_h, x), axis=1)
        })

        return self.h
    
    def output_layer(self):

        y_pred = self.preactivation_function(self.h, self.wy, self.by)
        return y_pred

    def backpropagation(self, y_pred, y_true, learning_rate):

        '''
        Docstring for backpropagation()

        :param y_true: this parameter should store actual values of response variable

        function is a hand realization of backpropagation for LSTM model
        '''
        # initialize matrixes to store derivatives for weights and biases
        dL_dwf = np.zeros((self.hidden_size + self.number_of_features, self.hidden_size))
        dL_dbf = np.zeros((1, self.hidden_size))
        dL_dwi = np.zeros((self.hidden_size + self.number_of_features, self.hidden_size))
        dL_dbi = np.zeros((1, self.hidden_size))
        dL_dwc = np.zeros((self.hidden_size + self.number_of_features, self.hidden_size))
        dL_dbc = np.zeros((1, self.hidden_size))
        dL_dwo = np.zeros((self.hidden_size + self.number_of_features, self.hidden_size))
        dL_dbo = np.zeros((1, self.hidden_size))
        
        # derivative of Loss with respect to output y: dL/dy = (y_pred - y_true)/batch_size
        # size is batch_size x lenght_output
        dL_dy = (y_pred - y_true)/self.batch_size

        # derivative of Loss with respect to weight wy: dL/dw = h * dL_dy
        # size is hidden_size x lenght_output
        dL_dwy = self.h.T @ dL_dy


        # derivative of Loss with respect to bias by: dL/db = sum(dL_dy) - summation over columns
        # size is 1 x length output
        dL_dby = np.sum(dL_dy, axis=0, keepdims=True)

        # derivative of Loss with respect to last short-term memory = dL_dh = wy * dL_dy
        # size batch_size x hidden_size
        dL_dh = dL_dy @ self.wy.T

        # initialize derivative of long-term memory coming from next step as zeroes
        dL_dc_next =  np.zeros((self.batch_size, self.hidden_size))
        # initialize derivative of short-term memory coming from next step
        dL_dh_next =  dL_dh


        for t in range(len(self.memory) - 1, -1, -1):

            # apply tahn function element_wise to long-term memory
            tahn_c_t = self.tahn_function(self.memory[t]["c"])

            # derivative of Loss function with respect to output gate: dL_do = tahn(c) * dh
            # size batch_size x hidden_size
            dL_do = dL_dh_next * tahn_c_t

            # derivative of Loss function with respect to long-term memory after update at time t
            # size batch_size x hidden_size
            dL_dc_t = dL_dh_next * self.memory[t]["o"] * (1 - (tahn_c_t ** 2))  + dL_dc_next

            # derivative of Loss function with respect to preactivated value of o
            dL_do_preact = (dL_do * self.memory[t]["o"] * (1 - self.memory[t]["o"]))

            # derivative of Loss function with respect to weight of output gate 
            # size is hidden+features x hidden
            dL_dwo += self.memory[t]["combined"].T @ dL_do_preact
            
            # derivative of Loss function with respect to bias of output gate 
            # size is 1 x hidden
            dL_dbo += np.sum(dL_do_preact, axis=0, keepdims=True)

            # derivative of Loss function with respect to active f
            # size batch_size x hidden_size
            dL_df = dL_dc_t * self.memory[t]["prev_c"]
            
            # derivative of Loss function with respect to active i
            # size batch_size x hidden_size
            dL_dct = dL_dc_t * self.memory[t]["i"]

            # derivative of Loss function with respect to active ct
            # size batch_size x hidden_size
            dL_di = dL_dc_t * self.memory[t]["ct"]

            # derivative of Loss function with respect to preactivated value of ct
            dL_dct_preact = dL_dct * (1 - (self.memory[t]["ct"] ** 2))
            # derivative of Loss function with respect to weight of ct gate 
            # size is hidden+features x hidden
            dL_dwc += self.memory[t]["combined"].T @ dL_dct_preact
            # derivative of Loss function with respect to bias of ct gate 
            # size is 1 x hidden
            dL_dbc += np.sum(dL_dct_preact, axis=0, keepdims=True)

            # derivative of Loss function with respect to preactivated value of i
            dL_di_preact = dL_di * self.memory[t]["i"] * (1 - self.memory[t]["i"])
            # derivative of Loss function with respect to weight of i gate 
            # size is hidden+features x hidden
            dL_dwi += self.memory[t]["combined"].T @ dL_di_preact
            # derivative of Loss function with respect to bias of ct gate 
            # size is 1 x hidden
            dL_dbi += np.sum(dL_di_preact, axis=0, keepdims=True)

            # derivative of Loss function with respect to preactivated value of i
            dL_df_preact = dL_df * self.memory[t]["f"] * (1 - self.memory[t]["f"])
            # derivative of Loss function with respect to weight of i gate 
            # size is hidden+features x hidden
            dL_dwf += self.memory[t]["combined"].T @ dL_df_preact
            # derivative of Loss function with respect to bias of ct gate 
            # size is 1 x hidden
            dL_dbf += np.sum(dL_df_preact, axis=0, keepdims=True)


            dL_dh_next = (dL_df_preact  @ self.wf.T +
                            dL_di_preact  @ self.wi.T +
                            dL_dct_preact @ self.wc.T +
                            dL_do_preact  @ self.wo.T)[:, :self.hidden_size]
            
            '''dL_dcombined = (dL_df_preact  @ self.wf.T +
                            dL_di_preact  @ self.wi.T +
                            dL_dct_preact @ self.wc.T +
                            dL_do_preact  @ self.wo.T)
            dL_dh_next = dL_dcombined[:, :self.hidden_size]'''
            # derivative of the c_t-1 that is passed to the previous iteration
            dL_dc_next = dL_dc_t * self.memory[t]["f"]


            ###################################### update weights and biases ###################################

        self.wc -= learning_rate * dL_dwc
        self.bc -= learning_rate * dL_dbc
        self.wf -= learning_rate * dL_dwf
        self.bf -= learning_rate * dL_dbf
        self.wi -= learning_rate * dL_dwi
        self.bi -= learning_rate * dL_dbi
        self.wo -= learning_rate * dL_dwo
        self.bo -= learning_rate * dL_dbo
        self.wy -= learning_rate * dL_dwy
        self.by -= learning_rate * dL_dby

    def training(self, X, X_val, scaler, learning_rate, num_epoches = 100):
        
        
        for epoch in range(num_epoches):

            # shaffle the data for each epoch to improve training
            indices = np.random.permutation(X.shape[0])
            X_shuffled = X.iloc[indices].reset_index(drop=True)
            indices = np.random.permutation(X_val.shape[0])
            X_val_shuffled = X_val.iloc[indices].reset_index(drop=True)

            train_losses = []        
            # iterate over the whole dataset with a step of a batch size
            for start in range(0, X.shape[0]-self.batch_size, self.batch_size):
                    
                batch_x = np.stack(X_shuffled["time_serie"].iloc[start:start+self.batch_size])
                batch_y = np.stack(X_shuffled["response"].iloc[start:start+self.batch_size])

                self.forward_sequence(batch_x)
                y_pred = self.output_layer()
                self.backpropagation(y_pred, batch_y, learning_rate)

                train_losses.append(np.mean((y_pred - batch_y) ** 2))

            val_losses = []
            for start in range(0, X_val.shape[0] - self.batch_size, self.batch_size):
                batch_x_val = np.stack(X_val_shuffled["time_serie"].iloc[start:start + self.batch_size])
                batch_y_val = np.stack(X_val_shuffled["response"].iloc[start:start + self.batch_size])

                self.forward_sequence(batch_x_val)        # forward only — no backprop
                y_val_pred = self.output_layer()
                val_losses.append(np.mean((y_val_pred - batch_y_val) ** 2))

            val_rmse = np.sqrt(np.mean(val_losses))
            train_rmse  = np.sqrt(np.mean(train_losses))

            epoch_rrmse = train_rmse / np.abs(np.mean(
            np.stack(X_shuffled["response"].values.tolist())  # full dataset mean
            ))

            print(f"Epoch {epoch + 1}/{num_epoches} — Train RMSE: {train_rmse*scaler.scale_[0]:.4f} Validation RMSE: {val_rmse*scaler.scale_[0]:.4f}  RRMSE: {epoch_rrmse:.4f}")

        