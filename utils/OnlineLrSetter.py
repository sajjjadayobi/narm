# alternative LROnPlateau 
import pandas as pd

def create_lrs(inits_lrs=[5e-3, 5e-3], file_name='lrs'):
  lrs = pd.DataFrame(data=inits_lrs, columns=['lrs'])
  lrs.index.name = 'epoch'
  lrs.to_csv(f'{file_name}.csv')

# we need add_lrs in another kernel for add during training
def add_lrs(path, lr):
  df = pd.read_csv(f'{path}.csv')
  df.append({'epoch': df.epoch.iloc[-1], 'lrs': lr}, ignore_index=True)
  df.to_csv(f'{path}.csv')

def scheduler(epoch):
  df = pd.read_csv('lrs.csv')
  return df.lrs[epoch]

callback = tf.keras.callbacks.LearningRateScheduler(schedule=scheduler, verbose=1)
