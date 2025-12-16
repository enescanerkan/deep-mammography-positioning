# early_stopping.py
import numpy as np
import torch

class EarlyStopping:
    """
    Validasyon kaybı belirli bir 'patience' süresi boyunca iyileşmediğinde eğitimi durdurur.
    """
    def __init__(self, patience=15, verbose=False, delta=0.001, path='checkpoint.pth', trace_func=print):
        """
        Args:
            patience (int): Validasyon kaybının iyileşmediği epoch sayısı.
            verbose (bool): True ise, her validasyon kaybı iyileştiğinde bir mesaj yazdırır.
            delta (float): Yeni en iyi modelin kabul edilmesi için validasyon kaybındaki minimum değişim.
            path (str): En iyi modelin kaydedileceği dosya yolu.
            trace_func (function): print fonksiyonu yerine kullanılacak fonksiyon.
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Validasyon kaybı azaldığında modeli kaydeder.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss
