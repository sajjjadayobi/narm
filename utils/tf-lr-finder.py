# use this trick for find lr range
# and do smaller range for next epoch
from tensorflow.keras.callbacks import Callback
import matplotlib.pyplot as plt

class LRFinderCallback(Callback):
    def __init__(self, min_lr, max_lr, mom=0.9, update_steps=3):
        self.min_lr = min_lr
        self.max_lr = max_lr 
        self.mom = mom # Make loss smoother using momentum
        self.update_steps = update_steps # update lr after 3 batch
        self.stop_multiplier = -20*self.mom/3 + 10
                
    def on_train_begin(self, logs={}):
        p = self.params
        n_iterations = p['steps']*p['epochs']    
        self.learning_rates = np.geomspace(self.min_lr, self.max_lr, num=n_iterations//self.update_steps+1)
        self.losses=[]
        self.iteration=0
        self.best_loss=0
    
    def on_batch_end(self, batch, logs={}):
        loss = logs.get('loss')
        if self.iteration!=0:
            loss = self.losses[-1]*self.mom+loss*(1-self.mom)
        if self.iteration==0 or loss < self.best_loss: 
                self.best_loss = loss
        if self.iteration%self.update_steps==0:
            lr = self.learning_rates[self.iteration//self.update_steps]            
            self.model.optimizer.lr = lr
            self.losses.append(loss)            

        if loss > self.best_loss*self.stop_multiplier: # Stop criteria
            self.model.stop_training = True         
        self.iteration += 1
    
    def on_train_end(self, logs=None):
        plt.figure(figsize=(12, 6))
        plt.plot(self.learning_rates[:len(self.losses)], self.losses)
        plt.xlabel("Learning Rate")
        plt.ylabel("Loss")
        plt.xscale('log')
        plt.show()
