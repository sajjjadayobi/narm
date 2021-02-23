import time, sys
import matplotlib.pylab as plt
import numpy as np
import torch

# order 1
class StatsAvg(Callback):
  # inner class  
  class AvgRunner():
    def __init__(self, mom=None):
        self.val, self.avg, self.sum, self.count = 0, 0, 0, 0

    def update(self, val, step=1):
        self.val = val
        self.sum += val * step
        self.count += step
        self.avg = self.sum / self.count

  # helper functions
  def __init__(self, mom=None):
      self.mom = mom

  def update_all(self, items, step):
      for avg, item in zip(self.metrcis_runners, items):
          avg.update(item, step)

  def get_avgs(self, names):
      return {n:a.avg for a, n in zip(self.metrcis_runners, names)}

  # valid section
  def on_valid_epoch_start(self):
    self.metrcis_runners = [self.AvgRunner(self.mom) for i in range(len(self.metrcis))]
    self.loss_runner = self.AvgRunner(self.mom)

  def on_valid_batch_end(self):
    self.loss_runner.update(self.params['loss'].item(), self.valid_batch)
    self.params['loss'] = self.loss_runner.avg
    self.update_all(self.params['metrics'].values(), self.valid_batch)
    self.params['metrics'] = self.get_avgs(self.params['metrics'].keys()) 
        
  def on_valid_epoch_end(self):
    del self.metrcis_runners
    del self.loss_runner

  # train section
  def on_train_epoch_start(self):
    self.metrcis_runners = [self.AvgRunner(self.mom) for i in range(len(self.metrcis))]
    self.loss_runner = self.AvgRunner(self.mom)

  def on_train_batch_end(self):
    self.loss_runner.update(self.params['loss'].item(), self.valid_batch)
    self.params['loss'] = self.loss_runner.avg
    self.update_all(self.params['metrics'].values(), self.valid_batch)
    self.params['metrics'] = self.get_avgs(self.params['metrics'].keys())     

  def on_train_epoch_end(self):
    del self.metrcis_runners
    del self.loss_runner

   
# order 2
class Reporter(Callback):
    # helper
    def __init__(self, after_step=1):
        self.after_step = after_step

    def show(self, phase='train'):
        report = '   loss: ' + str(round(self.params['loss'], 4))
        report += '  ' + str('   ').join([f"{k}: {round(v, 3)}" for k, v in self.params['metrics'].items()])
        sys.stdout.write(f"\r {phase}_steps: {self.params['step']+1}/{self.params[f'{phase}_steps']}" + report)
        sys.stdout.flush()
    
    # overrides
    def on_valid_epoch_end(self): 
        print('', end='\n') 

    def on_valid_batch_end(self): 
        if self.params['step']%self.after_step==0:
            self.show('valid')

    def on_train_batch_end(self):
        if self.params['step']%self.after_step==0:
            self.show('train')

        
    def on_train_epoch_start(self): 
        print('\n Epoch %2d/%2d' % (self.params['epoch'], self.params['epochs']))
        print('-' * 75)
        self.t0 = time.time()

    def on_train_epoch_end(self):
        t1 = time.time() - self.t0
        print('  time: %.0fm %.0fs' % (t1//60, t1%60))
                         
                         
# without order 
class Logger(Callback):
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
            plt.title('loss')
            plt.ylabel('loss')
            plt.xlabel('epoch')
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
              plt.xlabel('epoch')
              plt.legend(['train','valid'], loc='upper left')
              plt.show()
                         
                         
class LRFinder(Callback):
    def __init__(self, min_lr=1e-6, max_lr=5e-1, mom=0.9, update_steps=1, epoch=1):
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.epoch = epoch 
        self.mom = mom # Make loss smoother using momentum
        self.update_steps = update_steps # update lr after 3 batch
        self.stop_multiplier = -20*self.mom/3 + 10
                
    def on_train_start(self):
        self.init_weights = self.model.state_dict()
        n_iter = self.params['train_steps']*self.params['epochs']    
        self.learning_rates = np.geomspace(self.min_lr, self.max_lr, num=n_iter//self.update_steps+1)
        self.losses=[]
        self.best_loss=0

    def on_train_batch_end(self):
        loss = self.params['loss']
        step = self.params['step']
        if step!=0:
            loss = self.losses[-1]*self.mom+loss*(1-self.mom)
        if step==0 or loss < self.best_loss: 
            self.best_loss = loss
        if step%self.update_steps==0:
            lr = self.learning_rates[step//self.update_steps]            
            self.optimizer.lr = lr
            self.losses.append(loss)            


    def on_epoch_end(self):
        if self.params['epoch']==self.epoch:
            lr = min(self.losses)/10
            # show 
            print('\n')
            plt.figure(dpi=80)
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
