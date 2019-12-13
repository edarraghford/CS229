"""
Code to run final training on 250 image training set and calculate accuracy on 58 image test set 
Run after selecting hyper-parameters from cross-validation search
"""


import torch.nn as nn
import torch.optim
from torchvision import datasets, transforms
import torch.nn.functional as F
from kymatio import Scattering2D
import kymatio.datasets as scattering_datasets
import kymatio
import torch
import argparse
import math
import data 
import numpy as np 

class View(nn.Module):
    def __init__(self, *args):
        super(View, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view(-1,*self.shape)

#train model 
def train(model, device, train_loader, optimizer, epoch, scattering):
    model.train()
    correct = 0
    for batch_idx, (data, target) in enumerate(train_loader):
       
        data, target = data.to(device), target.to(device)
        data, target= torch.autograd.Variable(data), torch.autograd.Variable(target)        

        optimizer.zero_grad()
        output = model(scattering(data))
        loss = F.binary_cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        out = output.detach().numpy()
        targ = np.array(target)
        out[out >= 0.5] = 1
        out[out < 0.5] = 0
        pred = 1- np.abs(np.transpose(out)-targ)
        correct += np.sum(pred)
        if batch_idx % 4 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
    print('Model Accuracy: {}'.format(correct/len(train_loader.dataset)))

#test model 
def test(model, device, test_loader, scattering):
    model.eval()
    test_loss = 0
    correct_relaxed = 0
    correct_unrelaxed = 0 
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(scattering(data))
            test_loss += F.binary_cross_entropy(output, target) 
            out = np.array(output)
            targ = np.array(target)
            print(targ)
            print(out)
            out[out >= 0.5] = 1 
            out[out < 0.5] = 0 
            mask = targ == 1 
            out_relax = out[mask]
            targ_relax = targ[mask]
            pred_relax = 1- np.abs(np.transpose(out_relax)-targ_relax) 
            correct_relaxed += np.sum(pred_relax)  
            mask = targ == 0 
            out_unrelax = out[mask]
            targ_unrelax = targ[mask]
            pred_unrelax = 1- np.abs(np.transpose(out_unrelax)-targ_unrelax) 
            correct_unrelaxed += np.sum(pred_unrelax)
    test_loss /= len(test_loader.dataset)
    print(test_loss)
    print('\nTest set: Accuracy Relaxed: {}/{} ({:.2f}%), Accuracy Unrelaxed: {}/{} ({:.2f}%)\n'.format(
        correct_relaxed, len(targ_relax), correct_relaxed/len(targ_relax), correct_unrelaxed, len(targ_unrelax), correct_unrelaxed/len(targ_unrelax)))


def main():
    """
        Three models:
        'linear' - single-layer neural network
        'mlp' - multi-layer neural network 
        'cnn' - convolutional neural network

        scattering 1st order can also be set by the mode
        Scattering features are normalized by batch normalization.
    """
    parser = argparse.ArgumentParser(description='scattering + network analysis for cluster relaxation classification')
    parser.add_argument('--mode', type=int, default=2,help='scattering 1st or 2nd order')
    parser.add_argument('--classifier', type=str, default='cnn',help='classifier model')
    args = parser.parse_args()
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if args.mode == 1:
        scattering = Scattering2D(J=2, shape=(60, 60), max_order=1)
        K = 17
    else:
        scattering = Scattering2D(J=2, shape=(60, 60))
        K = 81
    if use_cuda:
        scattering = scattering.cuda()

    if args.classifier == 'cnn':
        model = nn.Sequential(
            View(K, 15, 15),
            nn.BatchNorm2d(K),
            nn.Conv2d(K, 64, 3, padding=1), nn.ReLU(),
            View(64*15*15),
            nn.Linear(64* 15 * 15, 512), nn.ReLU(),
            nn.Linear(512, 1), nn.Sigmoid()
        ).to(device)

    elif args.classifier == 'mlp':
        model = nn.Sequential(
            View(K, 15, 15),
            nn.BatchNorm2d(K),
            View(K*15*15),
            nn.Linear(K*15*15, 512), nn.ReLU(),
            nn.Linear(512, 512), nn.ReLU(),
            nn.Linear(512, 1), nn.Sigmoid()
        )

    elif args.classifier == 'linear':
        model = nn.Sequential(
            View(K * 15 * 15),
            nn.Linear(K *15 *15, 1), nn.Sigmoid()
        )

    # Load cluster data for train and test sets 
    if use_cuda:
        num_workers = 4
        pin_memory = True
    else:
        num_workers = None
        pin_memory = False
    batch_size = 16
    learning_rate = 0.0001
    print('batch size is: ' + str(batch_size))
    print('learning rate is: ' + str(learning_rate)) 
    train_loader = torch.utils.data.DataLoader(data.SimulationsDataset(image_file='obs_shuffled.txt', target_file = 'labels_shuffled.txt', train=0), batch_size=batch_size)
    test_loader = torch.utils.data.DataLoader(data.SimulationsDataset(image_file='obs_shuffled.txt', target_file = 'labels_shuffled.txt', train=1), batch_size=58)  
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=0.0005 , momentum = 0.9) 

    #initialize model 
    model.to(device)
    print("The model architecture is: ") 
    print(model)
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
            m.weight.data.normal_(0, 2./math.sqrt(n))
            m.bias.data.zero_()
        if isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 2./math.sqrt(m.in_features))
            m.bias.data.zero_()
    #train
    epochs = 14
    print("the number of training epochs will be: ")
    print(epochs) 
    for epoch in range(epochs):
        train( model, device, train_loader, optimizer, epoch, scattering)

    #test 
    test(model, device, test_loader, scattering)
if __name__ == '__main__':
    main()
