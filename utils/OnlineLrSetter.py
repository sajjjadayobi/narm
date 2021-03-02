# alternative LROnPlateau 
import pandas as pd
import tensorflow as tf

class OnlineLr():
  def __init__(self, init_lrs=[5e-3], path='lrs'):
      lrs = pd.DataFrame(data=init_lrs, columns=['lrs'])
      lrs.index.name = 'epoch'
      self.path = path+'.csv'
      lrs.to_csv(self.path)
      del lrs
  
  def add_lrs(self, lr):
      df = pd.read_csv(self.path)
      df.append({'epoch': df.epoch.iloc[-1], 'lrs': lr}, ignore_index=True)
      df.to_csv(f'{path}.csv')
      del df

  def __call__(self, epoch):
      df = pd.read_csv(self.path)
      return df.lrs[epoch]

scheduler = OnlineLr(init_lrs=[5e-5])
scheduler(0) # 5e-5
# using with fit
callback = tf.keras.callbacks.LearningRateScheduler(schedule=scheduler, verbose=1)
