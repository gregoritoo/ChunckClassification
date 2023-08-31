
import torch.nn.functional as F
from torch.utils.data import Dataset
import  torch.optim as torch_optim
import torch

class MatrixDataset(Dataset):
    def __init__(self, x_data, y_data):
        self.x_data = x_data
        self.y_data = y_data


    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        return self.x_data[idx],self.y_data[idx]



def train_model(model, optim, train_loader):
    model.train()
    total = 0
    sum_loss = 0
    for batch_x, batch_y in train_loader :
        optim.zero_grad()
        output = model(batch_x)
        loss = F.cross_entropy(output, batch_y,weight=torch.tensor([0.2,0.8]))   
        loss.backward()
        optim.step()
        total += batch_x.size(0)  
        sum_loss += batch_x.size(0) * loss.item() 
    return sum_loss / total

def val_loss(model, test_loader):
    model.eval()
    total = 0
    sum_loss = 0
    correct,correct_balanced = 0, 0
    for batch_x, batch_y in test_loader:
        current_batch_size = batch_y.size(0)
        output = model(batch_x)
        loss = F.cross_entropy(output, batch_y,weight=torch.tensor([0.2,0.8]))   
        sum_loss += current_batch_size * loss.item()
        total += current_batch_size
        pred = torch.max(output, 1)[1]
        correct += (pred == batch_y).float().sum().item()
    accuracy = correct / total
    print(f" validation loss {sum_loss/ total} validation balanced accuracy {accuracy}")
    return sum_loss / total, accuracy

def train_loop(model, epochs, train_loader, test_loader, lr=0.001, wd=0.0):
    loss_curve = []
    loss_validation = []
    optim = get_optimizer(model, lr=lr, wd=wd)
    scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=10, gamma=0.5)  
    for i in range(epochs): 
        loss = train_model(model, optim, train_loader)
        loss_curve.append(loss)
        loss_val, accuracy = val_loss(model, test_loader)
        loss_validation.append(loss_val)
        scheduler.step()  
    return loss_curve, loss_validation


def get_optimizer(model, lr = 0.001, wd = 0.0):
    optim = torch_optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    return optim