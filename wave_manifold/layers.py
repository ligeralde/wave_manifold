"""Module containing classes for the network layers: retina (input), lgn, hidden layer, and classifier.

  Typical usage example:

  retina = Retina()
  lgn = LGN()
"""

import torch
import numpy as np
from sklearn import svm
from sklearn.metrics import pairwise_distances
from sklearn.metrics import hinge_loss
import matplotlib.pyplot as plt

class Retina(object):
    def __init__(self, n_neurons=3200, grid_length=32):
        """Class for simulating a retina. Creates a grid of cells which can  
        produce simulated retinal waves. Grid can also be used to project
        real data (such as MNIST digits) onto. 
        ----------
        n_neurons : number of retinal ganglion cells in grid
        length : length of one side of square grid on which to place RGCs
        """
            
        self.n_neurons = n_neurons

        #get uniform random distributed positions in grid
        self.grid_length = grid_length
        self.pos = self.grid_length*np.random.rand(n_neurons, 2) 

        #rate for updating the voltage leak 
        self.a = 0.02*np.ones(n_neurons) 

        #weighting factor for voltage to update voltage leak
        self.b = 0.2*np.ones(n_neurons) 

        #get uniform random distributed reset voltages (value after firing)
        self.reset_voltage = -65+15*np.random.rand(n_neurons)**2
        
        #get centroid of cell grid
        centroid = np.mean(self.pos, axis=0)

        #get distance from each position to centroid 
        vect_from_cent = self.pos-centroid 
        dist_from_cent = np.linalg.norm(vect_from_cent - centroid, axis=-1) 

        #leak factor that's 2 at centroid and goes to 8 exponentially with distance from centroid 
        dist_falloff = 6*np.exp(-dist_from_cent/10) 
        self.distance_leak = 8 - dist_falloff 
        
        #distant cells (over 6 pixels) have exponentially decaying input 
        #nearby cells (under 2 pixels) have input of 5
        #remove diagonal elements (cell can't input into itself)
        self.dist_between_cells = pairwise_distances(self.pos) 
        self.proximity_weighted_gain = 5*(self.dist_between_cells < 2) - 2*(self.dist_between_cells > 6)*np.exp(-self.dist_between_cells/10) 
        self.proximity_weighted_gain -= np.diag(np.diag(self.proximity_weighted_gain)) 
                                                                                       
        #initialize voltages 
        self.voltage = -65*np.ones(n_neurons)

        #get voltage leak as percentage of initial voltage
        self.total_leak = self.b*self.voltage
        

    def prop_wave(self, return_fired_mask=True):
        """Updates internal variables for one time step of simulated retinal wave dynamics. 

        Args:
        return_fired_mask: optionally return mask of which neurons fired. 

        Returns:
        Binary mask corresponding to which neurons fired (1) or didn't (0)
        """

        #randomly initialize gain for each neuron
        self.total_gain = 3*np.random.randn(self.n_neurons)  
        
        #get indices mask of which neurons fired 
        fired = np.where(self.voltage >= 30)[0] 
        
        #optionally get binary vector of which neurons fired
        if return_fired_mask == True:
            fired_mask = (self.voltage >= 30).astype('int') 
        
        #only applies if neurons fired 
        if len(fired) != 0: 
            #reset the voltages for neurons that fired
            self.voltage[fired] = self.reset_voltage[fired]

            #get voltage leak with distance from centroid for neurons that fired
            self.total_leak[fired] += self.distance_leak[fired] 

            #get voltage gains based proximities of neurons that fired
            self.total_gain += np.sum(self.proximity_weighted_gain[:,fired],axis=1)

            #try gradually increasing the radius of activation? 
    
        #update voltage
        self.voltage += 0.04*self.voltage**2 + 5*self.voltage + 140 - self.total_leak + self.total_gain 

        #increase the voltage leak factor based on current voltage
        self.total_leak += self.a*(self.b*self.voltage-self.total_leak) 

        #optionally return firing mask
        if return_fired_mask == True:
            return(fired_mask)


class LGN(object):
    """Class for simulating the LGN. Creates a vector of cells which can  
        take input from the activities of a Retina object and output
        a vector of activations. 
        ----------
        n_neurons : number of LGN cells
        n_retinal_neurons : number of RGCs (size of input layer)
        weight_mu : (scalar) mean for weight initialization
        weight_mu : (scalar) std dev for weight initialization
        thresh_mu : (scalar) mean for threshold initialization
        thresh_mu : (scalar) std dev for threshold initialization
        device : string corresponding to name of GPU. defaults to None (CPU).
    """

    def __init__(self, 
                 n_neurons=400, 
                 n_retinal_neurons=3200, 
                 weight_mu=2.5, 
                 weight_sig=0.14, 
                 thresh_mu=70,
                 thresh_sig=2,
                 device=None):
        
        self.n_neurons = n_neurons
        self.weight_mu = weight_mu
        self.weight_sig = weight_sig

        #get normally distributed weights and normalize each lgn neuron synapses by its mean
        self.weights = torch.normal(weight_mu, weight_sig, (n_neurons, n_retinal_neurons), device=device) 
        self.weights = torch.div(self.weights,self.weights.mean(axis=1)[:,None]/weight_mu)
                                    
        #get normally distributed thresholds 
        self.thresholds = torch.normal(thresh_mu, thresh_sig, size=(n_neurons,), device=device)

        #initialize tracker of number of synaptic updates
        self.num_changes = torch.zeros(n_neurons, device=device)

        #initialize tracker of activities
        self.activations_history = torch.zeros(n_neurons, device=device)

        #optionally set GPU for pytorch
        self.device = device

    
    def pretrain(self, retina, timesteps=1000000, eta=0.1, plot_every=10000, track_every=None, data=None):
        """Pre-trains LGN weights (processing layer) using either simulated waves from 
        a Retina object instance or real retinal wave data. 

        Args:
        retina: instance of Retina object 
        """

        if track_every:
            track_acts = torch.zeros((track_every, self.n_neurons))
            track_intervals = timesteps//track_every 
        
        self.plot_rfs(retina)

        for t in range(timesteps):
            if data:
                #for training on real data images
                wave = data[:,t][:,None].type(torch.float)
            
            else:
                #for training with simulated retinal waves 
                wave = torch.tensor(retina.prop_wave())[:,None].type(torch.float)

            self.pretrain_step(wave, eta=eta)

            if track_every:
                if (t > 0 and t % track_intervals == 0):
                    track_acts[t//track_intervals, :] = self.max_pool_forward_pass(wave).squeeze()
                
            if (t > 0 and t % 1000 == 0): 
                self.update_thresholds()
                self.reset_activation_history()

            if t % plot_every == 0:
                plt.close()
                print('iteration: {}'.format(t))
                self.plot_rfs(retina)


    def pretrain_step(self, wave, eta):
        """Computes activations given retinal input and updates synapses w/ Hebbian learning.

        Args:
        wave: activities vector of length (n_retinal_neurons)
        eta: learning rate for synaptic update
        """

        #get activities vector (only one element > 0)
        activations = self.max_pool_forward_pass(wave).squeeze()
        
        #track number of synaptic changes (only occurs for one activated unit at each time step)
        self.num_changes[activations > 0] += 1

        #track activations  
        self.activations_history = torch.column_stack((self.activations_history, activations))
        
        #hebbian learning to update weights for activated unit
        if len(activations[activations > 0] > 0):
            self.weights[activations > 0, :] += 0.5*eta*torch.nan_to_num(torch.outer(activations[activations > 0], wave.squeeze()))

            #renormalize weights 
            self.weights[activations > 0, :] /= ((self.weights[activations > 0, :]).mean() / self.weight_mu)

            #increase threshold for activated unit
            self.thresholds[activations > 0] += 0.005*activations[activations>0]
        

    def max_pool_forward_pass(self, wave):
        """Computes winner-take-all activation vector given retinal input

        Args:
        wave: activities vector of length (n_retinal_neurons)

        Returns:
        activations: a vector of length (n_neurons) with only one nonzero element. 
        """

        #pass wave through synaptic weights
        weighted_inputs = self.lgn_forward_pass(self, wave)
        
        #get the activated unit
        activations = self.get_winner(self.ReLU(weighted_inputs))
        return activations


    def lgn_forward_pass(self, batch):
        return(torch.matmul(self.weights, batch))
        

    def get_winner(self, pre_activations):
        """Helper function to compute winner-take-all activations given pre-activations.

        Args:
        pre-activations: activities vector of length (n_neurons). Typically inputs that have been
        multiplied by the synaptic weights and passed through point-wise ReLU. 

        Returns:
        activations vector of length (n_neurons) with only one nonzero element. 
        """

        #threshold pre-activations and pass through ReLU
        acts_over_thresh = self.ReLU(pre_activations - self.thresholds[:,None])

        #get one hot vector corresponding to the max of the thresholded activations 
        max_act_one_hot = torch.nn.functional.one_hot(acts_over_thresh.max(axis=0).indices, num_classes=self.n_neurons).T.float()
        
        #return vector with the one nonzero element corresponding to the max
        return(torch.mul(max_act_one_hot, acts_over_thresh))
    

    def ReLU(self, inputs):
        """Helper function to compute pointwise ReLU outputs from a vector of inputs

        Args:
        inputs: 1-D tensor of any length. 

        Returns:
        1-D tensor of same length as input. 
        """
        return(torch.nn.functional.relu(inputs))
    

    def update_thresholds(self):
        """Lowers the thresholds for neurons that aren't changing much based on the max of their past activations. 
        """
        self.thresholds[self.num_changes < 200] = torch.max(self.activations_history[self.num_changes < 200,:],axis=1).values/5


    def reset_activation_history(self):
        """Shortens acitvation history by just maintaining a running maximum of past activations. 
        """
        self.activations_history = torch.max(self.activations_history,axis=1).values


    def plot_rfs(self, retina, idxs=None, mean_thresh=3, cmap='bwr', figsize=(12,12)):
        if idxs is not None:
            plot_len = np.sqrt(len(idxs)).astype('int')

        else:
            plot_len = np.sqrt(self.n_neurons).astype('int')
            idxs = np.arange(self.n_neurons)

        fig, axs = plt.subplots(n_rows=plot_len, n_cols=plot_len, figsize=figsize)

        for idx, ax in zip(idxs, axs.ravel()):
            rf = (self.weights[idx,:] - np.min(np.array(self.weights)))/np.max(np.array(self.weights))
            ax.scatter(retina.pos[:,0], retina.pos[:,1], c=rf, marker='.', cmap=cmap)
            ax.set_xlabel("")
            ax.set_xticklabels([])
            ax.set_xticks([])
            ax.set_yticklabels([])
            ax.set_yticks([])
        
        plt.show()



class FunctionalLGN(LGN):
    """Class that uses LGN as a base class. Utilizes LGN as a hidden layer
    that is appended to another hidden layer and output layer.  
        ----------
        n_hidden : number of hidden units in hidden layer
        n_output : number of units in output layer 
    """

    def __init__(self, 
                 n_output=10, 
                 n_hidden=1000, 
                 n_neurons=400, 
                 n_retinal_neurons=3200, 
                 weight_mu=2.5, 
                 weight_sig=0.14,
                 thresh_mu=70,
                 thresh_sig=2, 
                 svm_output=True,
                 device=None):
        
        #get properties of biological part 
        super().__init__(n_neurons=n_neurons, 
                         n_retinal_neurons=n_retinal_neurons, 
                         weight_mu=weight_mu, 
                         weight_sig=weight_sig,
                         thresh_mu=thresh_mu,
                         thresh_sig=thresh_sig,
                         device=device)

        #initialize hidden layer
        self.n_hidden = n_hidden 
        self.hidden_weights = torch.randn(n_hidden, self.n_neurons, device=device)
        
        #initialize output layer
        if svm_output == True:
            self.svm_classifier = svm.SVC()

        else:
            self.n_output = n_output
            self.output_weights = torch.randn(n_output, n_hidden, device=device)


    def hidden_forward_pass(self, batch, pool):
        """Pass a batch of data through the LGN and a hidden layer

        Args:
        batch: tensor of shape (n_retinal_neurons, batch size)
        pool: if true, do a max pool across all LGN neurons before forward passing to hidden layer

        Returns:
        activations vector of length (n_neurons, batch size) with only one nonzero element. 
        """
        
        if pool == True:
            return self.ReLU(torch.matmul(self.hidden_weights, self.ReLU(self.max_pool_forward_pass(batch))))
        
        else:
            return self.ReLU(torch.matmul(self.hidden_weights, self.ReLU(self.lgn_forward_pass(batch))))
    

    def train_svm(self, batch, targets, pool=False):
        """Train SVM output layer on hidden layer representations of batch. 
        Note: this function must convert the hidden layer reps to numpy arrays if given as tensors. 
        Note: SVM takes in data as (batch size, n_hidden) 

        Args:
        batch: tensor of shape (n_retinal_neurons, batch size)
        targets: tensor of shape (batch size)
        """

        hidden_rep = self.hidden_forward_pass(batch, pool)
        self.svm_classifier.fit(np.array(hidden_rep.T), np.array(targets))
    

    def evaluate_svm_performance(self, batch, targets, pool=False):
        """ Computes hinge loss and accuracy on batch/targets 

        Args:
        batch: tensor of shape (n_retinal_neurons, batch size)
        targets: tensor of shape (batch size)

        Returns:
        tuple of loss and accuracy (scalars)
        """

        #get targets and hidden layer representation
        targets = np.array(targets)
        hidden_rep = self.hidden_forward_pass(batch, pool)

        #predict on hidden layer representation
        predicted_targets = self.svm_classifier.predict(np.array(hidden_rep.T))

        #evaluate loss and accuracy
        loss = hinge_loss(targets, predicted_targets)
        acc = self.get_svm_accuracy(targets, predicted_targets)
        return(loss, acc)


    def get_svm_accuracy(self, predicted_targets, true_targets):
        """
        Args: 
        predicted_targets: 1-D array with dim batch size
        true_targets: 1-D array with dim batch size
        """
        return np.sum(predicted_targets == true_targets)/len(true_targets)
    
    #code for training output layer; matrix inversion doesn't work well (low accuracy)
    # def train_output_layer(self, batch, labels, pool=True):
    #     """Analytically calculate the optimal weights for the output layer given training batch/labels.

    #     Args:
    #     batch: tensor of shape (n_retinal_neurons, batch size)
    #     labels: one-hot tensor of shape (num_classes, batch size) 
    #     """

    #     hidden_output = self.hidden_forward_pass(batch, pool=pool)
    #     self.output_weights = torch.matmul(torch.matmul(labels, hidden_output.T), torch.inverse(torch.matmul(hidden_output, hidden_output.T)))


    # def predict_labels(self, batch, pool, onehot):
    #     """Computes full forward pass through the whole network to get output (predicted labels) on batch of data.

    #     Args:
    #     batch: tensor of shape (n_retinal_neurons, batch size)
    #     onehot: (T/F) convert outputs to one hot vectors
    #     """

    #     network_out = torch.matmul(self.output_weights, self.hidden_forward_pass(batch, pool=pool))
    
    #     if onehot == True:
    #         return torch.nn.functional.one_hot(network_out.max(axis=0).indices, num_classes=self.n_output).T
                
    #     else: 
    #         return network_out
    

    # def get_accuracy(self, batch, labels, pool=True, onehot=True):
    #     """Gets accuracy of predicted labels on batch of data.
    #     """
    #     prediction = self.predict_labels(batch, pool, onehot)
    #     return(torch.div(torch.sum(torch.mul(labels, prediction)),batch.shape[1]))
            
