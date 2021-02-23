# ofiical
import torch
import time, sys
from torchsummary import summary

# utils
from callback import CallbackRunner
from callbacks import Averager, Averagers


class Trainer:
    def __init__(self, model, train_ds, valid_ds, train_batch=32, valid_batch=64, optimizer=None,
                 loss=None, scheduler=None, scheduler_type='epoch', metrcis=[], workers=4, fp16=False):
        
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scheduler_type = scheduler_type
        self.loss = loss
        self.metrcis = metrcis
        self.fp16 = fp16
        self.scaler = torch.cuda.amp.GradScaler(enabled=fp16)
        self.device = None
        self.workers = workers
        # prepare dls
        self.train_ds, self.valid_ds = train_ds, valid_ds
        self.train_batch = train_batch  
        self.valid_batch = valid_batch
        self.train_dl = self.get_train_dataloder(train_ds, train_batch)
        self.valid_dl = self.get_valid_dataloder(valid_ds, valid_batch)
        # logs 
        self.stop = False
        self.params = {'train_steps': len(self.train_dl), 'valid_steps': len(self.valid_dl)}
    


    # dataloders
    def get_train_dataloder(self, ds, batch_size, shuffle=True): 
        return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=self.workers)      
      
    def get_valid_dataloder(self, ds, batch_size, shuffle=False): 
        return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=self.workers)

            
    # faster trainig
    def change_device(self, device='cuda'):
        self.model.to(device)
        self.loss.to(device)
        self.device = device

    # metrics
    def compute_metrics(self, pred, y):
      res = {}
      for f in self.metrcis:
        res[f.__name__] = f(pred, y).item()
      return res
        
    # training 
    def fit(self, epochs=1, device='cuda', callbacks=[]):
        self.params['epochs'] = epochs
        self.change_device(device)
        self.state = CallbackRunner(callbacks, trainer=self)

        self.state.on_train_start
        for epoch in range(1, epochs+1):
            self.params['epoch'] = epoch
            self.state.on_epoch_start
            # train
            self.state.on_train_epoch_start
            self.train_epoch()
            self.state.on_train_epoch_end
            if self.scheduler != None and self.scheduler_type == 'epoch':
                self.scheduler.step()
            # valid
            self.state.on_valid_epoch_start
            self.valid_epoch()
            self.state.on_valid_epoch_end    
            self.state.on_epoch_end
        self.state.on_train_end

        
    def train_epoch(self):
      self.model.train()
      for step, batch in enumerate(self.train_dl, 0):
          self.params['step'] = step
          self.optimizer.zero_grad()
          batch = [item.to(self.device) for item in batch]
          self.params['batch'] = batch
          self.state.on_train_batch_start
             
          if self.device == 'cuda': # GPU + FP16
            with torch.cuda.amp.autocast(enabled=self.fp16):
                loss, metrics = self.train_step(batch)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                if self.scheduler != None and self.scheduler_type == 'batch':
                    self.scaler.step(self.scheduler)
                self.scaler.update()
          else: # CPU
              loss, metrics = self.train_step(batch)
              loss.backward()
              self.optimizer.step()
              if self.scheduler != None and self.scheduler_type == 'batch':
                  self.scheduler.step()

          # stats
          self.params['loss'] = loss
          self.params['metrics'] = metrics
          self.state.on_train_batch_end

      return None # use for stop

    
    def valid_epoch(self):
        self.model.eval()
        with torch.no_grad():
            for step, batch in enumerate(self.valid_dl, 0):
              self.params['step'] = step
              batch = [item.to(self.device) for item in batch]
              self.params['batch'] = batch
              self.state.on_valid_batch_start

              if self.device == 'cuda': # GPU + FP16
                  with torch.cuda.amp.autocast(enabled=self.fp16):
                      self.params['loss'], self.params['metrics'] = self.valid_step(batch)
              else:
                self.params['loss'], self.params['metrics'] = self.valid_step(batch)
              
              self.state.on_valid_batch_end
        
        return None

    #@overrider
    def train_step(self, batch):
        x, y = batch
        out = self.model(x)
        loss = self.loss(out, y)
        # self.log('train_loss', loss)
        metrcis = self.compute_metrics(out, y)
        return loss, metrcis

    #@overrider
    def valid_step(self, batch):
        x, y = batch
        out = self.model(x)
        loss = self.loss(out, y)
        metrcis = self.compute_metrics(out, y)
        return loss, metrcis


    # ----------------------------------------------------------------------------------- UTILS
    def summary(self):
        summary(self.model, input_size=(self.train_ds[0][0].shape))

     # plotters
    def set_defualt(self): pass
    def lr_finder(self): pass
    def loss_plot(self): pass
    def metrcis_plot(self): pass
    def most_worse(self): pass
    # freeze
    def freeze(self): pass
    def unfreaze(self): pass
    # checkpoint
    def save(self): pass
    def load(self): pass
    def resume(self): pass
    # infernce
    def predict(self): pass
    def evaluate(self): pass
    # expriments
    def repuodicible(self): pass 
