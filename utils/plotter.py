from IPython import display
import matplotlib.pyplot as plt
import tensorflow as tf

# plot online loss or accuracy
class OnlinePlotter:
  def __init__(self, sec, xlabel='', ylabel=''):
    self.history = []
    self.xlabel = xlabel
    self.ylabel = ylabel
    self.sec = sec
    self.tic = time.time()

  def plot(self, data):
    self.history.append(data)
    plt.cla()
    plt.plot(self.history)
    plt.xlabel(self.xlabel) 
    plt.ylabel(self.ylabel)
    display.clear_output(wait=True)
    display.display(plt.gcf())
    
    
class RuningLoss:
  def __init__(self, smoothing_factor=0.0):
    self.alpha = smoothing_factor
    self.loss = []
    
  def append(self, value):
    self.loss.append( self.alpha*self.loss[-1] + (1-self.alpha)*value if len(self.loss)>0 else value )
  def get(self):
    return self.loss    
    
    
def tf_display_model(model):
  tf.keras.utils.plot_model(model, to_file='tmp.png', show_shapes=True)
  return display.Image('tmp.png')
