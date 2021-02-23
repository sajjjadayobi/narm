class Callback():
    def set_trainer(self, trainer):
        self.trainer = trainer

    def __getattr__(self, attr):
        return getattr(self.trainer, attr)

    # fit
    def on_train_start(self): pass
    def on_train_end(self): pass
    def on_epoch_start(self): pass
    def on_epoch_end(self): pass
    # train
    def on_train_epoch_start(self): pass
    def on_train_epoch_end(self): pass
    def on_train_batch_start(self): pass
    def on_train_batch_end(self): pass
    # valid
    def on_valid_epoch_start(self): pass
    def on_valid_epoch_end(self): pass
    def on_valid_batch_start(self): pass
    def on_valid_batch_end(self): pass


class CallbackRunner():
    def __init__(self, callbacks, trainer):
        self.callbacks = callbacks
        for cb in self.callbacks:
          cb.set_trainer(trainer)

    def __getattr__(self, attr):
        for cb in self.callbacks:
          getattr(cb, attr)()
        return None
