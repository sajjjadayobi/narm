from IPython import display
import matplotlib.pyplot as plt
import time

# plot online loss or accuracy
class OnlinePlotter:
  # plotter = OnlinePlotter(1, 'iterations', 'loss/acc', .9)
  # plotter.plot(loss/acc)
  def __init__(self, sec=2, xlabel='', ylabel='', mom=0.0, scale=None):
    self.scale = scale # type of plots
    self.mom = mom # for runninig loss
    self.sec = sec # plot in every 2 sec
    self.history = []
    self.xlabel = xlabel
    self.ylabel = ylabel
    self.tic = time.time()

  def plot(self, data):
    self.history.append(self.mom*self.history[-1] + (1-self.mom)*data if len(self.history)>0 else data)
    if time.time() - self.tic > self.sec:
      plt.cla()
      if self.scale == 'semilogy':
          plt.semilogy(self.history)
      elif self.scale == 'loglog':
          plt.loglog(self.history)
      elif self.scale == None:
          plt.plot(self.history)

      plt.xlabel(self.xlabel) 
      plt.ylabel(self.ylabel)
      display.clear_output(wait=True)
      display.display(plt.gcf())
      self.tic = time.time()
