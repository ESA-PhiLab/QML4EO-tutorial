## Reference: https://stackoverflow.com/questions/71998978/early-stopping-in-pytorch

## Solution 1 
## Alessandro Sebastianelli -> I added abs(validation_loss - train_loss) since it is possible to have negative values
class EarlyStopping1:
    '''
    Usage:
    
    early_stopping = EarlyStopping(patience=3, min_delta=10)
    for epoch in np.arange(n_epochs):
        train_loss = train_one_epoch(model, train_loader)
        validation_loss = validate_one_epoch(model, validation_loader)
        
        early_stopping(train_loss, validation_loss)
        if early_stopping.early_stop:             
            break
    '''
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.early_stop = False

    def __call__(self, train_loss, validation_loss):
        if abs(validation_loss - train_loss) >= self.min_delta:
            self.counter +=1
            if self.counter >= self.patience:  
                self.early_stop = True

## Solution 2 
## Alessandro Sebastianelli -> I adjusted the functions to have a similar structure with solution 1
class EarlyStopping2:
    # Although @KarelZe's response solves your problem sufficiently and elegantly, I want to provide an alternative early stopping 
    # criterion that is arguably better. Your early stopping criterion is based on how much (and for how long) the validation loss 
    # diverges from the training loss. This will break when the validation loss is indeed decreasing but is generally not close enough
    # to the training loss. The goal of training a model is to encourage the reduction of validation loss and not the reduction in
    # the gap between training loss and validation loss. Hence, I would argue that a better early stopping criterion would be watch 
    # for the trend in validation loss alone, i.e., if the training is not resulting in lowering of the validation loss then terminate it. 
    '''
    Usage:

    early_stopper = EarlyStopper(patience=3, min_delta=10)
    for epoch in np.arange(n_epochs):
        train_loss = train_one_epoch(model, train_loader)
        validation_loss = validate_one_epoch(model, validation_loader)
        if early_stopper.early_stop(validation_loss):             
            break
    '''
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf
        self.early_stop = False

    def __call__(self, train_losses, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
