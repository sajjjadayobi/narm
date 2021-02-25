# ofiical
import torch
import time, sys
from torchsummary import summary
from copy import deepcopy

# utils
from callbacks import CallbackRunner, LRFinder


# ofiical
import torch
import time, sys
from torchsummary import summary
from copy import deepcopy
from torch.nn.utils import clip_grad_norm_



class Trainer:
    def __init__(self, model, train_ds, valid_ds=None, train_bs=32, valid_bs=64, optimizer=None, loss=None, 
                 scheduler=None, scheduler_type='epoch', metrcis=[], workers=4, fp16=False, grad_clip_value='inf'):
        
        # without valid
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scheduler_type = scheduler_type
        self.loss = loss
        self.metrcis = metrcis
        self.grad_clip_value = grad_clip_value
        self.fp16 = fp16
        self.scaler = torch.cuda.amp.GradScaler(enabled=fp16)
        self.workers = workers
        self.stop_training = False # check
        # prepare dls
        self.train_bs = train_bs
        self.valid_bs = valid_bs
        self.train_ds = train_ds
        self.valid_ds = valid_ds 
        self.train_dl = self.get_train_dataloder()
        self.params = {'train_steps': len(self.train_dl)}
        if valid_ds==None:
          self.valid_dl = None
        else:
          self.valid_dl = self.get_valid_dataloder()
          self.params['valid_steps'] = len(self.valid_dl)
    

            
    # faster trainig
    def change_device(self, device='cuda'):
        self.model.to(device)
        self.loss.to(device)
        self.device = device
        self.optimizer.load_state_dict(self.optimizer.state_dict())


    # metrics
    def compute_metrics(self, pred, y):
      res = {}
      for f in self.metrcis:
        res[f.__name__] = f(pred, y).item()
      return res
        
    # training 
    def fit(self, epochs=1, device='cuda', callbacks=[]):
        if self.stop_training: return
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
            if self.valid_dl!=None:
                self.state.on_valid_epoch_start
                self.valid_epoch()
                self.state.on_valid_epoch_end    
              
            self.state.on_epoch_end
            if self.stop_training: 
              break
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
                clip_grad_norm_(self.model.parameters(), self.grad_clip_value)
                self.scaler.step(self.optimizer)
                if self.scheduler != None and self.scheduler_type == 'batch':
                    self.scaler.step(self.scheduler)
                self.scaler.update()
          else: # CPU
              loss, metrics = self.train_step(batch)
              loss.backward()
              clip_grad_norm_(self.model.parameters(), self.grad_clip_value)
              self.optimizer.step()
              if self.scheduler != None and self.scheduler_type == 'batch':
                  self.scheduler.step()

          # stats
          self.params['loss'] = loss
          self.params['metrics'] = metrics
          self.state.on_train_batch_end
          if self.stop_training: 
              break

    
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
              if self.stop_training: 
                break
        
        

    # -------------------------------------------------------------------------------- overrider
    def train_step(self, batch):
        x, y = batch
        out = self.model(x)
        loss = self.loss(out, y)
        # self.log('train_loss', loss)
        metrcis = self.compute_metrics(out, y)
        return loss, metrcis

    def valid_step(self, batch):
        x, y = batch
        out = self.model(x)
        loss = self.loss(out, y)
        metrcis = self.compute_metrics(out, y)
        return loss, metrcis

    def get_train_dataloder(self): 
        return torch.utils.data.DataLoader(self.train_ds, batch_size=self.train_bs, shuffle=True, num_workers=self.workers)      
      
    def get_valid_dataloder(self): 
        return torch.utils.data.DataLoader(self.valid_ds, batch_size=self.valid_bs, shuffle=False, num_workers=self.workers)


    # -------------------------------------------------------------------------------- utils
    def summary(self):
        summary(self.model, input_size=(self.train_ds[0][0].shape), device='cpu')


    def lr_finder(self, device='cuda', epochs=1, min_lr=1e-6, max_lr=1e1):
        init_weights = deepcopy(model.state_dict())
        # off valid epoch
        valid_dl = self.valid_dl
        self.valid_dl = None
        self.fit(epochs=epochs, device=device, callbacks=[LRFinder(min_lr=min_lr, max_lr=max_lr)])
        # reset training
        self.model.load_state_dict(init_weights)
        self.valid_dl = valid_dl


    def set_lr(self, lr):
        self.optimizer.param_groups[0]['lr'] = lr
      

    
    def most_worse(self): pass
    # freeze
    def freeze(self): pass
    def unfreaze(self): pass
    # checkpoint
    def save(self): pass
    def load(self): pass
    def resume(self): pass
    # infernce
    def evaluate(self): pass
    # expriments
    def repuodicible(self): pass 
