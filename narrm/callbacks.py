import time, sys
import matplotlib.pylab as plt
import numpy as np
import torch

class Reporter(Callback):
    # helper
    def stats2str(self):
        loss = 'loss: ' + str(round(self.params['loss'], 4))
        metrcis = str('   ').join([f"{k}: {round(v, 3)}" for k, v in self.params['metrics'].items()])
        return '  ' + loss + '  ' + metrcis
    
    # overrides
    def on_valid_epoch_end(self): 
        print('', end='\n') 

    def on_valid_batch_end(self): 
        sys.stdout.write(f"\r Valid_Steps: {self.params['step']+1}/{self.params['valid_steps']}" + self.stats2str())
        sys.stdout.flush()

    def on_train_batch_end(self):
        sys.stdout.write(f"\r Train_Steps: {self.params['step']+1}/{self.params['train_steps']}" + self.stats2str())
        sys.stdout.flush()

    def on_train_epoch_start(self): 
        print('\n Epoch %2d/%2d' % (self.params['epoch']+1, self.params['epochs']))
        print('-' * 75)
        self.t0 = time.time()

    def on_train_epoch_end(self):
        t1 = time.time() - self.t0
        print('  time: %.0fm %.0fs' % (t1//60, t1%60))
        
    

class LoggerPlotter(Callback):
  def __init__(self, plot_loss=True, plot_metrics=True):
      self.plot_loss = plot_loss
      self.plot_metrics = plot_metrics

  def on_train_start(self):
        self.train_metrcis = {f.__name__:[] for f in self.metrcis}
        self.valid_metrcis = {f.__name__:[] for f in self.metrcis}
        self.valid_losses = []
        self.train_losses = []      
    
  def on_valid_epoch_end(self): 
        self.valid_losses.append(self.params['loss'])
        for k, v in self.params['metrics'].items():
            self.valid_metrcis[k].append(v)

  def on_train_epoch_end(self):
        self.train_losses.append(self.params['loss']) 
        for k, v in self.params['metrics'].items():
            self.train_metrcis[k].append(v)
    
  def on_train_end(self):
        if self.plot_loss:
            print('\n')
            plt.figure(dpi=80)
            plt.plot(self.train_losses)
            plt.plot(self.valid_losses)
            plt.title('model losses')
            plt.ylabel('loss')
            plt.xlabel('steps')
            plt.legend(['train','valid'], loc='upper left')
            plt.show()
        if self.plot_metrics:
            for k, v in self.valid_metrcis.items():
              print('\n')
              plt.figure(dpi=80)
              plt.plot(self.train_metrcis[k])
              plt.plot(v)
              plt.title(k)
              plt.ylabel('value')
              plt.xlabel('steps')
              plt.legend(['train','valid'], loc='upper left')
              plt.show()
 

class LRFinder(Callback):
    def __init__(self, min_lr, max_lr, mom=0.9, update_steps=1, epoch=1):
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.epoch = epoch 
        self.mom = mom # Make loss smoother using momentum
        self.update_steps = update_steps # update lr after 3 batch
        self.stop_multiplier = -20*self.mom/3 + 10
                
    def on_train_start(self):
        self.init_weights = self.model.state_dict()
        n_iterations = self.params['train_steps']*self.params['epochs']    
        self.learning_rates = np.geomspace(self.min_lr, self.max_lr, num=n_iterations//self.update_steps+1)
        self.losses=[]
        self.iteration=0
        self.best_loss=0
    
    def on_train_batch_end(self):
        loss = self.params['loss']
        if self.iteration!=0:
            loss = self.losses[-1]*self.mom+loss*(1-self.mom)
        if self.iteration==0 or loss < self.best_loss: 
                self.best_loss = loss
        if self.iteration%self.update_steps==0:
            lr = self.learning_rates[self.iteration//self.update_steps]            
            self.optimizer.lr = lr
            self.losses.append(loss)            
        if loss > self.best_loss*self.stop_multiplier:
            self.model.stop_training = True         
        self.iteration += 1

    def on_epoch_end(self):
        if self.params['epoch']==self.epoch:
            lr = min(self.losses)/10
            # show 
            print('\n')
            plt.figure(figsize=(12, 6))
            plt.plot(self.learning_rates[:len(self.losses)], self.losses)
            plt.title(f'I think we find it: {lr}')
            plt.xlabel("Learning Rate")
            plt.ylabel("Loss")
            plt.xscale('log')
            plt.show()
            # set for rest training
            self.optimizer.lr = lr
            self.model.load_state_dict(self.init_weights)
            for cb in self.state.callbacks:
              if cb == self:
                print('\n Training from Start with new LR (LRFinder removed from Callbacks) \n')
                del cb
