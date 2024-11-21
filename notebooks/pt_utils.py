import matplotlib.pyplot as plt
from tqdm import trange
import numpy as np
import torch


def sample_batch(dataset):
    batch = dataset.take(1).get_single_element()
    if isinstance(batch, tuple):
        batch = batch[0]
    return batch.numpy()


def display(
    images, n=10, size=(20, 3), cmap="gray_r", save_to=None
):
    """
    Displays n random images from each one of the supplied arrays.
    """
    if images.max() > 1.0:
        images = images / 255.0
    elif images.min() < 0.0:
        images = (images + 1.0) / 2.0

    plt.figure(figsize=size)
    for i in range(n):
        _ = plt.subplot(1, n, i + 1)
        plt.imshow(images[i].permute(1, 2, 0), cmap=cmap)
        plt.axis("off")

    if save_to:
        plt.savefig(save_to)
        print(f"\nSaved to {save_to}")

    plt.show()


class Trainer():
    def __init__(self, model, optimizer, train_loader, val_loader, loss_fn, pred_fn=None, device='cuda', metrics=[]):
        self.model = model
        self.device = device
        self.metrics = metrics
        self.loss_fn = loss_fn
        self.pred_fn = pred_fn if pred_fn else lambda logits: torch.argmax(logits, dim=-1)
        self.optimizer = optimizer
        self.val_loader = val_loader
        self.train_loader = train_loader
    
    def fit(self, epochs):

        history = { 'val_loss': [] }
        if 'train_loss' in self.metrics:
            history['train_loss'] = []
        if 'accuracy' in self.metrics:
            history['accuracy'] = []


        with trange(epochs, desc='Epochs', unit='epoch', leave=True) as pbar:
            for epoch in (range(epochs)):

                train_loss, val_loss, correct = 0, 0, 0

                # train pass
                self.model.train()
                for tb, (inputs, labels) in enumerate(self.train_loader):
                    inputs, labels = inputs.to(self.device), labels.to(self.device)

                    self.optimizer.zero_grad()
                    logits = self.model(inputs)
                    loss = self.loss_fn(logits, labels)
                    loss.backward()
                    self.optimizer.step()
                    train_loss += loss.item()


                # validation pass
                self.model.eval()
                with torch.no_grad():
                    for vb, (inputs, labels) in enumerate(self.val_loader):                
                        inputs, labels = inputs.to(self.device), labels.to(self.device)
                        logits = self.model(inputs)
                        loss = self.loss_fn(logits, labels)
                        val_loss += loss.item()

                        if self.pred_fn and 'accuracy' in self.metrics:
                            correct += torch.sum(self.pred_fn(logits) == labels).item()

                val_loss /= (vb + 1)
                train_loss /= (tb + 1)
                accuracy = correct/len(self.val_loader.dataset)

                history['val_loss'].append(val_loss)
                postfix_str = f'Loss: {val_loss:0.4f}'

                if 'train_loss' in self.metrics:
                    history['train_loss'].append(train_loss)
                    postfix_str += f', Training Loss: {train_loss:0.4f}'

                if 'accuracy' in self.metrics:
                    history['accuracy'].append(accuracy)
                    postfix_str += f', Accuracy: {accuracy:0.4f}'
                
                pbar.update()
                pbar.set_postfix_str(postfix_str)
        
        return history

    def evaluate(self, test_loader):

        val_loss, correct = 0, 0

        # validation pass
        self.model.eval()
        with torch.no_grad(), trange(len(test_loader.dataset), desc='Evaluating', unit='sample', leave=True) as pbar:
            for vb, (inputs, labels) in enumerate(self.val_loader):                
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                logits = self.model(inputs)
                loss = self.loss_fn(logits, labels)
                val_loss += loss.item()

                if self.pred_fn and 'accuracy' in self.metrics:
                    correct += torch.sum(self.pred_fn(logits) == labels).item()
                
                pbar.update(len(inputs))

        val_loss /= (vb + 1)
        accuracy = correct/len(self.val_loader.dataset)
        
        history = {'val_loss' : val_loss}
        if 'accuracy' in self.metrics:
            history['accuracy'] = accuracy
    
        return list(history.values())


    def predict(self, data):

        preds = []

        # validation pass
        self.model.eval()

        with torch.no_grad():
            if isinstance(data, torch.utils.data.DataLoader):
                loader = data
                with trange(len(loader.dataset), desc='Predicting', unit='sample', leave=True) as pbar:
                    for vb, (inputs, _) in enumerate(loader):                
                        inputs = inputs.to(self.device)
                        preds.extend(self.pred_fn(self.model(inputs)).detach().cpu().numpy())
                        pbar.update(len(inputs))
            else:
                with trange(len(data), desc='Predicting', unit='sample', leave=True) as pbar:
                    for inputs in (inputs.unsqueeze(0) for inputs in data):
                        inputs = inputs.to(self.device)
                        preds.extend(self.pred_fn(self.model(inputs)).cpu().numpy())
                        pbar.update()
    
        return np.array(preds)

