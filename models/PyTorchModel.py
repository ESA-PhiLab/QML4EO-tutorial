from utils.DatasetHandler import DatasetHandler

from sklearn.metrics import confusion_matrix, classification_report
from utils.utils import print_CF
import matplotlib.pyplot as plt
from torchinfo import summary
from tqdm.auto import tqdm
import torch

class PyTorchModel:

    def __init__(self, model, criterion, optimizer):
        self.model      = model
        self.criterion  = criterion
        self.optimizer  = optimizer

        self.tra_losses = [0] + []
        self.val_losses = [0] + []

        self.dh = DatasetHandler(None)

    def summary(self, shape):
        print(summary(self.model, shape))
    
    def curves(self):
        fig = plt.figure()
        plt.plot(self.tra_losses[1:], '-*', label = 'training')
        plt.plot(self.val_losses[1:], '-*', label = 'validation')
        plt.legend()
        plt.show()
        
    def fit(self, epochs, tra_set, val_set, batch_size=1, es=None, classes = classes, tra_size = None, val_size = None):

        tra_imgs = tra_set[0]
        tra_lbls = tra_set[1]

        val_imgs = val_set[0]
        val_lbls = val_set[1]

        if tra_size == None: tra_size = len(tra_imgs)
        if val_size == None: val_size = len(val_imgs)

        pb = tqdm(range(epochs), dynamic_ncols=True)
        esct = 0

        for epoch in pb:
            tra_dl  = iter(self.dh.data_loader(tra_imgs, tra_lbls, batch_size=batch_size, img_shape=(64,64,3)))
            val_dl  = iter(self.dh.data_loader(val_imgs, val_lbls, batch_size=batch_size, img_shape=(64,64,3)))
            
            ## Training
            tra_loss = 0.0
            val_loss = 0.0
            for i in range(tra_size//batch_size):
                x, y    = next(tra_dl)
                outputs = self.model.forward(x)
                self.optimizer.zero_grad()
                loss    = self.criterion(outputs, y) 
                loss.backward() 
                self.optimizer.step() 

                l = loss.item()
                tra_loss += l
                pb.set_description("Train - [E %d/%d s %d] [B %d/%d - B Loss %1.4f] - T Loss %1.4f - V Loss %1.4f" % (epoch+1, epochs, esct+1, i+1, tra_size//batch_size, l, self.tra_losses[-1], self.val_losses[-1]))
            
            self.tra_losses.append(tra_loss/i)
            
            ## Validation
            pb2 = tqdm(range(val_size//batch_size), leave=False, dynamic_ncols=True)
            with torch.no_grad():
                targets = []
                predictions = []

                for i in pb2:
                    x, y    = next(val_dl)

                    outputs = self.model.forward(x)
                    loss    = self.criterion(outputs, y)

                    targets.append(y.item())
                    predictions.append(self.model.predict(x).item())

                    l = loss.item()
                    val_loss += l

                    pb2.set_description("Valid - [E %d/%d s %d] [B %d/%d - B Loss %1.4f] - T Loss %1.4f - V Loss %1.4f" % (epoch+1, epochs, esct+1, i+1, val_size//batch_size, l, self.tra_losses[-1], self.val_losses[-1]))
                
                self.val_losses.append(val_loss/i)
            
            ## Early Stopping
            if es!=None:
                es(self.tra_losses[-1], self.val_losses[-1])
                if es.early_stop:
                    break
                esct = es.counter

            if epoch % 3 == 1:
                cf = confusion_matrix(targets, predictions, normalize='true')
                cr = classification_report(targets, predictions, target_names=classes, digits=4)

                print('\t\t [*] Confusion Matrix:')
                print_CF(cf, classes)
                print('\t\t [*] Classification Report:')
                print(cr)