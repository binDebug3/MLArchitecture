# Class dependencies
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import minimize
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import pickle


# Other analysis libraries
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation


"""
To import this class into an ipynb file in the same folder:

from model import acme    
"""



class acme():
    def __init__(self, max_iter=1000, tol=1e-8, learning_rate=0.01, reg=10, dim_reg=0, 
                 optimizer="grad", batch_size=32, epochs=100, random_state=None, **kwargs):
        """Initialize the Adaptive Covariance Metric Estimator (ACME) model
        
        Parameters:
            max_iter (int) - The maximum number of iterations to perform in the gradient descent optimization
            tol (float) - The tolerance for the amount of improvement between iterations during gradient descent optimization
            learning_rate (float) - The learning rate for the gradient descent optimization
            reg (float) - The regularization parameter for learning the covariance matrix
            dim_reg (float) - The regularization parameter for the dimension of the covariance matrix
                            This regularizer adds the identity matrix scaled by dim_reg to the covariance matrix to penalize
                            the model for dropping dimensions
            optimizer (str) - The optimizer to use. 
                            Options are 'grad' for gradient descent, 'sgd' for stochastic gradient descent, and 'bfgs' for BFGS
            batch_size (int) - The batch size for stochastic gradient descent
            epochs (int) - The number of epochs for stochastic gradient descent
            random_state (int) - The random state for the model
            **kwargs - Additional parameters for the model

        Attributes:
            X (n,d) ndarray - The data to fit the model on
            y (n,) ndarray - The labels of the data
            n (int) - The number of data points
            d (int) - The dimension of the data
            weights (d,d) ndarray - The weights to learn
            classes (num_classes,) ndarray - The classes of the data
            num_classes (int) - The number of classes
            y_dict (dict) - A dictionary mapping the labels to integers
            one_hot (n,num_classes) ndarray - The one hot encoding of the labels
            differences (n,n,d) ndarray - The differences array for the informative points
            cur_gaussian (n,n) ndarray - The current gaussian kernel for the informative points
            cur_tensor_prod (n,n,d) ndarray - The current tensor product for the informative points
            subset_differences (list) - A list of differences arrays for each subset
            X_val_set (n_val,d) ndarray - The validation set for the data
            y_val_set (n_val,) ndarray - The validation labels for the data
            class_index (list) - A list of arrays of the indices of each class in the labels
                                List of length num_classes
                                Elements are arrays of about length n/num_classes
            val_history (list) - A list of validation accuracies
            train_history (list) - A list of training accuracies
            weights_history (list) - A list of the weights at each iteration
        Returns:
            None
        """
        # Hyperparameters
        self.max_iter = max_iter
        self.tol = tol
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.epochs = epochs
        self.reg = reg
        self.dim_reg = dim_reg
        self.random_state = random_state
        np.random.seed(self.random_state)
        self.__dict__.update(kwargs)

        # Variables to store
        self.X = None
        self.y = None
        self.n = None
        self.d = None
        self.min_iter = 5
        self.save_frequency = 100
        self.weights = None
        self.classes = None
        self.num_classes = None
        self.y_dict = None
        self.one_hot = None
        self.differences = None
        self.cur_gaussian = None
        self.cur_tensor_prod = None
        self.subset_differences = []

        # Validation variables
        self.X_val_set = None
        self.y_val_set = None
        self.class_index = []
        self.val_history = []
        self.train_history = []
        self.weights_history = []
        self.can_validate = False
        

    ############################## Helper Functions ###############################
    def set_params(self, **kwargs):
        """
        Set the parameters of the model

        Parameters:
            **kwargs - The parameters to set
        Returns:
            None
        """
        # TODO: Test this function
        for key, value in kwargs.items():
            try:
                setattr(self, key, value)
            except Exception as e:
                print(f"Could not set {key} to {value}. Error: {e}")
    

    def get_params(self):
        """
        Get the parameters of the model

        Parameters:
            None
        Returns:
            params (dict) - The parameters of the model
        """
        # TODO: Test this function
        return self.__dict__
    

    def label_from_stationary(self, stationary:np.ndarray, show_probabilities=False):
        """
        Get the label for each class from the stationary distribution.

        Parameters:
            stationary (n,) ndarray - The stationary distribution of the data
            show_probabilities (bool) - Whether to return the probabilities of the classes
        Returns:
            labels (n,) ndarray - The predicted labels of the data 
            OR
            class_probabilities (num_classes,) ndarray - The probabilities of each class based on the stationary distribution
        """
        # Runtime check to ensure `stationary` is a 1D ndarray
        if not isinstance(stationary, np.ndarray):
            raise TypeError("stationary must be a numpy ndarray.")
        if stationary.ndim != 1:
            raise ValueError("stationary must be a 1D ndarray.")
        # Check if the sum of the stationary distribution is approximately 1
        if not np.isclose(np.sum(stationary), 1):
            raise ValueError("The weights do not sum to 1.")
        if not type(show_probabilities) == bool:
            raise TypeError("show_probabilities must be a boolean.")
        
        # Sum the weights for each class
        class_probabilities = np.zeros(self.num_classes)
        for weight, label in zip(stationary, self.y):
            class_probabilities[self.y_dict[label]] += weight

        # Return the probabilities if requested
        if show_probabilities:
            return class_probabilities
        
        # Otherwise, return the class with the highest weight
        else:
            indices = np.argmax(class_probabilities)
            return self.classes[indices]


    def encode_y(self, y:np.ndarray):
        """
        Encode the labels of the data.

        Parameters:
            y (n,) ndarray - The labels of the data
        Returns:
            None
        """
        # Check if the input is a list
        if isinstance(y, list):
            y = np.array(y)

        # Make sure it is a numpy array
        elif not isinstance(y, np.ndarray):
            raise ValueError("y must be a list or a numpy array")
        
        # If it is not integers, give it a dictionary
        if y.dtype != int:
            self.classes = np.unique(y)
            self.y_dict = {label: i for i, label in enumerate(np.unique(y))}

        # If it is, still make it a dictionary
        else:
            self.classes = np.arange(np.max(y)+1)
            self.y_dict = {i: i for i in self.classes}
        self.num_classes = len(self.classes)

        # Create an array of the y indices for each class respectively
        for i in range(self.num_classes):
            self.class_index.append(np.where(y == self.classes[i])[0])

        # Make a one hot encoding
        self.one_hot = np.zeros((self.n, self.num_classes))
        for i in range(self.n):
            self.one_hot[i, self.y_dict[y[i]]] = 1

        
    def randomize_batches(self):
        """
        Randomize the batches for stochastic gradient descent
        Parameters:
            None
        Returns:
            batches (list) - A list of batches of indices for training
        """
        # Get randomized indices and calculate the number of batches
        indices = np.arange(self.n)
        np.random.shuffle(indices, random_state=self.random_state)
        num_batches = self.n // self.batch_size

        # Loop through the different batches and get the batches
        batches = [indices[i*self.batch_size:(i+1)*self.batch_size].tolist() for i in range(num_batches)]

        # Handle the remaining points
        remaining_points = indices[num_batches*self.batch_size:]
        counter = len(remaining_points)
        i = 0

        # Fill the remaining points into the batches
        while counter > 0:
            batches[i % len(batches)].append(remaining_points[i])
            i += 1
            counter -= 1

        # Return the batches
        return batches


    ############################## Training Calculations ##############################
    def update_differences(self, X:np.ndarray, batches=None):
        """
        Get the array of difference between the informative points
        
        Parameters:
            X (n,d) ndarray - The data to calculate the differences for
            batches (list) - A list of batches of indices for training
            NOT?
            informative_points (n,d) ndarray - The informative points
        Returns:
            None
            NOT?
            differences (n,n,d) ndarray - The differences array for the informative points
        """
        # If it is not a batch, calculate the differences
        if batches is None:
            self.differences = X[:,np.newaxis,:] - X[np.newaxis,:,:]
        
        # Otherwise, calculate the differences for each batch
        else:
            self.subset_differences = [X[batch][:,np.newaxis,:] - X[batch][np.newaxis,:,:] for batch in batches]


    
    def update_gaussian(self, weights:np.ndarray, subset_num=None):
        """
        Get the gaussian kernel for the informative points
        
        Parameters:
            weights (d,) ndarray - The weights for the informative points
            NOT?
            informative_points (n,d) ndarray - The informative points
            target (n,) ndarray - The target values
        Returns:
            gaussian (n,n) ndarray - The gaussian kernel for the informative points"""
        # If there is no subset, let the differences be the self.differences
        if subset_num is None:
            differences = self.differences
        else:
            differences = self.subset_differences[subset_num]

        # Calculate the gaussian kernel and the tensor product, and save them
        tensor_prod = np.einsum('ijk,lk->ijl', differences, weights)
        self.cur_gaussian = np.exp(-np.linalg.norm(tensor_prod, axis=2)).T
        self.cur_tensor_prod = tensor_prod



    def gradient(self, W:np.ndarray, subset=None, subset_num=None):
        """
        Compute the gradient of the loss function

        Parameters:
            W (d,d) ndarray - The weights for the informative points
            subset (list) - A list of indices for the subset
            subset_num (int) - The number of the subset
        Returns:
            dW (d,d) ndarray - The gradient of the loss function
        """

        """
        CHAT GPT optimization, most notably the tensor_prod and class_index replacing y_index
        # Initialize the gradient
        self.update_gaussian(W, subset_num)
        dW = np.zeros((self.d, self.d))

        # Use subset-specific differences or all differences
        differences = self.differences if subset is None else self.subset_differences[subset_num]
        one_hot = self.one_hot if subset is None else self._get_one_hot(subset)

        # Calculate gradient for each class
        for i, class_index in enumerate(self.classes):
            g_c = self.cur_gaussian[class_index]
            product_c = self.cur_tensor_prod[class_index]
            differences_c = differences[class_index]

            # Calculate weighted sum for the class
            weighted_product_c = g_c[:, :, np.newaxis] * product_c
            weighted_sum_c = np.sum(np.einsum('ijk,ijl->ijkl', weighted_product_c, differences_c), axis=0)
            weighted_sum_c /= np.sum(g_c, axis=0)[:, np.newaxis, np.newaxis] + 1e-20

            # Update gradient
            dW += np.sum(one_hot[:, i][:, np.newaxis, np.newaxis] * weighted_sum_c, axis=0)

        # Calculate the gradient second term
        g_all_totals = np.sum(self.cur_gaussian, axis=0)[:, np.newaxis, np.newaxis] + 1e-20
        weighted_product_all = self.cur_gaussian[:, :, np.newaxis] * self.cur_tensor_prod
        weighted_sum_all = np.sum(np.einsum('ijk,ijl->ijkl', weighted_product_all, differences), axis=0)
        weighted_sum_all /= g_all_totals

        # Update gradient
        dW -= np.sum(weighted_sum_all, axis=0)

        # Regularize the gradient
        reg_term = self.reg * (W / np.linalg.norm(W, 'fro') - self.dim_reg * np.eye(self.d) / self.d)
        return 2 * dW + reg_term
        """
        # Initialize the gradient
        self.update_gaussian(W, subset_num)
        dW = np.zeros((self.d, self.d))
        differences = self.differences if subset is None else self.subset_differences[subset_num]

        # If there is no subset, let the differences be the self.differeces and the one hot be the self.one_hot
        if subset is None:
            y_index = self.class_index
            one_hot = self.one_hot
        
        # Otherwise, select the subset
        else:
            y_index = []
            y_sub = self.y[subset]

            # Loop through the different classes and select the right subsets
            for i in range(self.num_classes):
                y_index.append(np.where(y_sub == self.classes[i])[0])

            # Modify the one hot encoding to just the subset
            one_hot = np.zeros((len(subset), self.num_classes))
            for i in range(len(subset)):
                one_hot[i, self.y_dict[y_sub[i]]] = 1

        # Loop through the different classes and select the right subsets
        for i in range(len(self.classes)):
            g_c = self.cur_gaussian[y_index[i]]
            g_c_totals = np.sum(g_c, axis=0)[:, np.newaxis, np.newaxis] + 1e-20
            product_c = self.cur_tensor_prod[y_index[i]]
            differences_c = differences[y_index[i]]

            # Calculate the weighted products
            weighted_product_c = g_c[:,:,np.newaxis] * product_c
            weighted_sum_c = np.sum(np.einsum('ijk,ijl->ijkl', weighted_product_c, differences_c), axis=0)
            weighted_sum_c /= g_c_totals

            # Calculate the gradient first term
            dW += np.sum(one_hot[:,i][:,np.newaxis, np.newaxis] * weighted_sum_c, axis=0)

        # Calculate the gradient first term
        g_all_totals = np.sum(self.cur_gaussian, axis=0)[:, np.newaxis, np.newaxis] + 1e-20
        weighted_product_all = self.cur_gaussian[:,:,np.newaxis] * self.cur_tensor_prod
        weighted_sum_all = np.sum(np.einsum('ijk,ijl->ijkl', weighted_product_all, differences), axis=0)
        weighted_sum_all /= g_all_totals

        # Calculate the gradient second term
        dW -= np.sum(weighted_sum_all, axis=0)

        # Return the regularized gradient
        return 2*dW + self.reg*(W / np.linalg.norm(W, 'fro') - self.dim_reg*np.eye(self.d) / self.d)
    


    def loss(self, W:np.ndarray, subset=None, subset_num=None):
        """
        Compute the loss function

        Parameters:
            W (d,d) ndarray - The weights for the informative points
            subset (list) - A list of indices for the subset
            subset_num (int) - The number of the subset
        Returns:
            loss (float) - The loss value
        """
        # Initialize the loss and total gaussian
        self.update_gaussian(W, subset_num)
        loss = 0
        total_g_log = np.log(np.sum(self.cur_gaussian, axis=0) + 1e-20)

        # Get the right y index
        if subset is None:
            y_index = self.class_index
        else:
            y_index = []
            for i in range(self.num_classes): # For each class
                y_index.append(np.where(self.y[subset] == self.classes[i])[0]) # Get the indices of the class within the subset
        
        # Loop through the different classes and select the right subsets
        for i in range(len(self.classes)):
            g_c = self.cur_gaussian[y_index[i]]
            loss += np.log(np.sum(g_c, axis=0) + 1e-20)

        # Calculate the loss
        return np.sum(loss - total_g_log) + self.reg*(np.linalg.norm(W, 'fro') - self.dim_reg*np.trace(W) /self.d)



    ########################## Optimization and Training Functions ############################

    def gradient_descent(self):
        """
        Perform gradient descent on the model
        Parameters:
            None
        Returns:
            None
        """
        # TODO: check optimization changes I made
        iter_denom = 50
        show_iter = max(self.max_iter // self.save_frequency, self.min_iter) 
        break_tol = self.tol * self.n
        iter_min = self.max_iter // iter_denom
        progress = tqdm(range(self.max_iter), desc="Processing classes", dynamic_ncols=True)

        for i in progress:
            # Get the gradient
            gradient = self.gradient(self.weights)
            self.weights -= self.learning_rate * gradient

            # If there is a validation set, check the validation error
            if i % show_iter == 0:
                self.weights_history.append(self.weights.copy())
                if self.can_validate:
                    # Predict on the validation set and append the history
                    val_accuracy = accuracy_score(self.y_val_set, self.predict(self.X_val_set))
                    self.val_history.append(val_accuracy)                    
                    progress.set_description(f"Iter {i}: Val Accuracy: {np.round(val_accuracy, 5)}")

            # Check for convergence after a certain number of iterations
            if i > iter_min and np.linalg.norm(gradient) < break_tol:
                break
        progress.close()



    def stochastic_gradient_descent(self, re_randomize=False):
        """
        Perform stochastic gradient descent on the model

        Parameters:
            re_randomize (bool) - Whether to re-randomize the batches after each epoch
                                    Should always be True but it doesn't work right now
        Returns:
            None
        """
        # Raise an error if there are no epochs or batch size, or if batch size is greater than the number of points
        if self.batch_size is None or self.epochs is None:
            raise ValueError("Batch size or epochs must be specified")
        if self.batch_size > self.n:
            raise ValueError("Batch size must be less than the number of points")
        
        # Initialize the loop, get the batches, and go through the epochs
        show_iter = max(self.epochs // self.save_frequency, self.min_iter) 
        val_accuracy = 0

        # NOTE: I removed the initial randomize and update differences
        loop = tqdm(total=self.epochs * len(self.n // self.batch_size), position=0)
        for epoch in range(self.epochs):

            # reset the batches if re_randomize is true
            if re_randomize:
                batches = self.randomize_batches()
                self.update_differences(self.X, batches)
            
            # Loop through the different batches
            for i, batch in enumerate(batches):

                # Get the gradient, update weights, and append the loss
                gradient = self.gradient(self.weights, subset=batch, subset_num=i)
                self.weights -= self.learning_rate * gradient
                curr_loss = self.loss(self.weights, subset=batch, subset_num=i)

                # update our loop
                loop.set_description('epoch:{}, loss:{:.4f}, val acc:{:.4f}'.format(epoch, curr_loss, val_accuracy))
                loop.update(1)

            if epoch % show_iter == 0:
                # Append the history of the weights
                self.weights_history.append(self.weights.copy())
                # If there is a validation set, check the validation error
                if self.can_validate:
                    val_accuracy = accuracy_score(self.y_val_set, self.predict(self.X_val_set))
                    self.val_history.append(val_accuracy)
        loop.close()



    def bfgs(self):
        """
        Perform Broyden-Fletcher-Goldfarb-Shanno (BFGS) optimization on the model

        Parameters:
            None
        Returns:
            None
        """
        # Define iterations to show the progress and define the loss and gradient function in 1D
        show_iter = max(self.max_iter // self.save_frequency, self.min_iter) 
        loss_bfgs = lambda W: self.loss(W.reshape(self.d, self.d))
        gradient_bfgs = lambda W: self.gradient(W.reshape(self.d, self.d)).flatten()
        self.bfgs_count = 0
        self.bfgs_val_accuracy = 0

        # Define the callback function
        def callback(weights):
            self.bfgs_count += 1
            self.weights = weights.reshape(self.d, self.d)
            progress.update(1)

            if self.bfgs_count % show_iter == 0:
                self.weights_history.append(self.weights.copy())

                # If there is a validation set, check the validation error
                if self.can_validate:
                    self.bfgs_val_accuracy = accuracy_score(self.y_val_set, self.predict(self.X_val_set))
                    self.val_history.append(self.bfgs_val_accuracy)
                progress.set_description(f"Iter {self.bfgs_count}: Val Accuracy: {np.round(self.bfgs_val_accuracy, 5)}.")

        # Run the optimizer
        progress = tqdm(total=self.max_iters, desc="BFGS Progress", dynamic_ncols=True)
        res = minimize(loss_bfgs, self.weights.flatten(), jac=gradient_bfgs, method='BFGS', 
                       options={'disp': False, 'maxiter': self.max_iter, 'gtol':self.tol}, 
                       callback=callback)
        self.weights = res.x.reshape(self.d, self.d)



    def fit(self, X:np.ndarray, y:np.ndarray, X_val_set=None, y_val_set=None, init_weights=None):
        """
        Fit the model to the data

        Parameters:
            X (n,d) ndarray - The data to fit the model on
            y (n,) ndarray - The labels of the data
            X_val_set (n_val,d) ndarray - The validation set for the data
            y_val_set (n_val,) ndarray - The validation labels for the data
            init_weights (d,d) ndarray - The initial weights for the model
        Returns:
            train_history (list) - A list of training accuracies
            val_history (list) - A list of validation accuracies
        """
        # Save the data as variables and encode y
        self.X = np.array(X)
        self.y = np.array(y)
        self.n = X.shape[0]
        self.d = X.shape[1]
        self.encode_y(y)

        # Initialize the weights
        if init_weights is not None:
            self.weights = init_weights
        else:
            self.weights = .125*((np.random.random((self.d, self.d))*2 - 1) + np.eye(self.d))

        # If there is a validation set, save it
        if X_val_set is not None and y_val_set is not None:
            self.X_val_set = X_val_set
            self.y_val_set = y_val_set

        # Perform necessary differences calculations
        if self.optimizer != "sgd":
            self.update_differences(self.X)

        # Run the optimizer
        if self.optimizer == "sgd":
            self.stochastic_gradient_descent()
        elif self.optimizer == "bfgs":
            self.bfgs()
        elif self.optimizer == "grad":
            self.gradient_descent()

        # Otherwise, raise an error
        else:
            raise ValueError("Optimizer must be 'sgd', 'bfgs, or 'grad'")
        
        return self.train_history, self.val_history



    ############################## Prediction Functions #############################

    def predict(self, points:np.ndarray, show_probabilities=False):
        """
        Predict the labels of the data
        
        Parameters:
            points (n,d) ndarray - The data to predict
        Returns:
            predictions (n,) ndarray - The predicted labels of the data
        """
        # Check the shape of the data and the point and initialize the predictions
        if len(points.shape) == 1:
            points = points[np.newaxis,:]
        predictions = []

        # Get the differences array
        differences = self.X[:,np.newaxis,:] - points[np.newaxis,:,:]
        probs = np.exp(-np.linalg.norm(np.einsum('ijk,lk->ijl', differences, self.weights), axis=2)).T + 1e-75
        probs /= np.sum(probs, axis=1, keepdims=True)
        
        # Loop through the different points and get the predictions
        for i in range(points.shape[0]):
            predictions.append(self.label_from_stationary(probs[i], show_probabilities=show_probabilities))

        # Return the predictions
        return np.array(predictions)
    


    def score(self, X:np.ndarray=None, y:np.ndarray=None):
        """
        Get the accuracy of the model on the data
        
        Parameters:
            X (n,d) ndarray - The data to score the model on
            y (n,) ndarray - The labels of the data
        Returns:
            accuracy (float) - The accuracy of the model on the data
        """
        # If the data is not provided, use the training data
        if X is None:
            X = self.X
            y = self.y

        # TODO: Test this function
        # Get the predictions and return the accuracy
        return accuracy_score(y, self.predict(X))
    

    def cross_val_score(self, X:np.ndarray, y:np.ndarray, cv=5):
        """
        Get the cross validated accuracy of the model on the data
        
        Parameters:
            X (n,d) ndarray - The data to score the model on
            y (n,) ndarray - The labels of the data
            cv (int) - The number of cross validation splits
        Returns:
            scores (list) - The accuracy of the model on the data for each split
        """
        #TODO: Test this function
        # Split the data and initialize the scores
        scores = []
        for train_index, test_index in train_test_split(np.arange(X.shape[0]), test_size=1/cv):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            # Fit the model and get the score
            self.fit(X_train, y_train)
            scores.append(self.score(X_test, y_test))
        
        # Return the scores
        return scores
    

    def confusion_matrix(self, X:np.ndarray=None, y:np.ndarray=None):
        """
        Get the confusion matrix of the model on the data
        
        Parameters:
            X (n,d) ndarray - The data to get the confusion matrix for
            y (n,) ndarray - The labels of the data
        Returns:
            confusion_matrix (num_classes, num_classes) ndarray - The confusion matrix of the model
        """
        #TODO: Test this function
        # If the data is not provided, use the training data
        if X is None:
            X = self.X
            y = self.y

        # Get the predictions and return the confusion matrix
        predictions = self.predict(X)
        return confusion_matrix(y, predictions, labels=self.classes)


    ############################## Other Functions ###############################
    def copy(self):
        """
        Create a copy of the model

        Parameters:
            None
        Returns:
            acme (class object) - A copy of the model
        """
        # Initialize a new model
        new_model = acme(max_iter=self.max_iter, tol=self.tol, learning_rate=self.learning_rate, 
                           reg=self.reg, dim_reg=self.dim_reg, 
                           optimizer=self.optimizer, batch_size=self.batch_size, epochs=self.epochs)
        
        # Save the attributes of the new model
        new_model.X = self.X
        new_model.y = self.y
        new_model.n = self.n
        new_model.d = self.d
        new_model.weights = self.weights
        new_model.classes = self.classes
        new_model.num_classes = self.num_classes
        new_model.y_dict = self.y_dict
        new_model.one_hot = self.one_hot
        new_model.differences = self.differences
        new_model.cur_gaussian = self.cur_gaussian
        new_model.cur_tensor_prod = self.cur_tensor_prod
        new_model.X_val_set = self.X_val_set
        new_model.y_val_set = self.y_val_set
        new_model.class_index = self.class_index
        new_model.val_history = self.val_history
        new_model.train_history = self.train_history
        new_model.weights_history = self.weights_history

        # Return the new model
        return new_model
    


    def save_weights(self, file_path:str, save_type="standard"):
        """
        Save the weights of the model to a file so that it can be loaded later

        Parameters:
            file_path (str) - The name of the file to save the weights to
            save_type (str) - How much of the model to save
                "full" - Save the full model and all of its attributes
                "standard" - Save the standard attributes of the model
                "weights" - Save only the weights of the model
        Returns:
            None
        """
        # TODO: Test this function
        if save_type not in ["full", "standard", "weights"]:
            raise ValueError("save_type must be 'full', 'standard', or 'weights'")
        
        preferences = {"weights": self.weights,
                       "save_type": save_type}
        if save_type == "standard":
            standar_preferences = {"max_iter": self.max_iter, 
                        "tol": self.tol, 
                        "learning_rate": self.learning_rate,
                        "optimizer": self.optimizer,
                        "batch_size": self.batch_size,
                        "epochs": self.epochs,
                        "reg": self.reg,
                        "dim_reg": self.dim_reg, 
                        "n": self.n,
                        "d": self.d,
                        "classes": self.classes,
                        "num_classes": self.num_classes,
                        "y_dict": self.y_dict,
                        "one_hot": self.one_hot,
                        }
            preferences.update(standar_preferences)
        
        if save_type == "full":
            remaining_attributes = {"X": self.X,
                                    "y": self.y,
                                    "differences": self.differences,
                                    "cur_gaussian": self.cur_gaussian,
                                    "cur_tensor_prod": self.cur_tensor_prod,
                                    "subset_differences": self.subset_differences,
                                    "X_val_set": self.X_val_set,
                                    "y_val_set": self.y_val_set,
                                    "class_index": self.class_index,
                                    "val_history": self.val_history,
                                    "train_history": self.train_history,
                                    "weights_history": self.weights_history,
                                    }
            preferences.update(remaining_attributes)

        try:
            with open(f'{file_path}.pkl', 'wb') as f:
                pickle.dump(preferences, f)
        except Exception as e:
            print(e)
            raise ValueError(f"The file '{file_path}.pkl' could not be saved.")
    


    def load_weights(self, file_path):
        """
        Load the weights of the model from a file

        Parameters:
            file_path (str) - The name of the file to load the weights from
        Returns:
            None
        """
        # TODO: Test this function
        try:
            with open(f'{file_path}.pkl', 'rb') as f:
                data = pickle.load(f)
            save_type = data["save_type"]

            self.weights = data["weights"]
            if save_type == "standard" or save_type == "full":
                self.max_iter = data["max_iter"]
                self.tol = data["tol"]
                self.learning_rate = data["learning_rate"]
                self.optimizer = data["optimizer"]
                self.batch_size = data["batch_size"]
                self.epochs = data["epochs"]
                self.reg = data["reg"]
                self.dim_reg = data["dim_reg"]
                self.n = data["n"]
                self.d = data["d"]
                self.classes = data["classes"]
                self.num_classes = data["num_classes"]
                self.y_dict = data["y_dict"]
                self.one_hot = data["one_hot"]
            if save_type == "full":
                self.X = data["X"]
                self.y = data["y"]
                self.differences = data["differences"]
                self.cur_gaussian = data["cur_gaussian"]
                self.cur_tensor_prod = data["cur_tensor_prod"]
                self.subset_differences = data["subset_differences"]
                self.X_val_set = data["X_val_set"]
                self.y_val_set = data["y_val_set"]
                self.class_index = data["class_index"]
                self.val_history = data["val_history"]
                self.train_history = data["train_history"]
                self.weights_history = data["weights_history"]

        except Exception as e:
            print(e)
            raise ValueError(f"The file '{file_path}.pkl' could not be loaded")
        

    
    def __str__(self):
        """
        Get the string representation of the model

        Parameters:
            None
        Returns:
            string (str) - The string representation of the model
        """
        # TODO: Test this function
        key_width = max(len(key) for key in self.__dict__.keys()) + 2

        print("\n\nAdaptive Covariance Metric Estimator (ACME) Model\n")
        for key, value in self.__dict__.items():
            if isinstance(value, np.ndarray):
                print(f"{key:<{key_width}} shape: {value.shape}")
            elif isinstance(value, list):
                print(f"{key:<{key_width}} length: {len(value)}")
            elif isinstance(value, dict):
                print(f"{key:<{key_width}} keys: {list(value.keys())}")
            else:
                print(f"{key:<{key_width}}: {value}")