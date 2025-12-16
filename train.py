import torch 
import os
from tqdm import trange
import argparse
from torchvision import datasets, transforms
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


from model import Generator, SAN_Discriminator, GAN_Discriminator
from utils import SAN_D_train, D_train, SAN_G_train, G_train, save_models
from visualize import visualize_san




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train GAN.')
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of epochs for training.")
    parser.add_argument("--lr", type=float, default=0.0002,
                      help="The learning rate to use for training.")
    parser.add_argument("--batch_size", type=int, default=64, 
                        help="Size of mini-batches for SGD")
    parser.add_argument("--model", type=str, default='SAN',
                        help="Type of Discriminator : GAN or SAN")

    args = parser.parse_args()


    os.makedirs('chekpoints', exist_ok=True)
    os.makedirs('data', exist_ok=True)

    # Data Pipeline
    print('Dataset loading...')
    # MNIST Dataset
    transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5), std=(0.5))])

    train_dataset = datasets.MNIST(root='data/MNIST/', train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root='data/MNIST/', train=False, transform=transform, download=False)


    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                               batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                              batch_size=args.batch_size, shuffle=False)
    print('Dataset Loaded.')


    print('Model Loading...')
    mnist_dim = 784
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = "mps" if torch.backends.mps.is_available() else device

    if args.model == 'GAN':
        G = torch.nn.DataParallel(Generator(g_output_dim = mnist_dim)).to(device)
        D = torch.nn.DataParallel(GAN_Discriminator(mnist_dim)).to(device)
    elif args.model == 'SAN':
        G = torch.nn.DataParallel(Generator(g_output_dim = mnist_dim)).to(device)
        D = torch.nn.DataParallel(SAN_Discriminator(mnist_dim)).to(device)


    # model = DataParallel(model).cuda()
    print('Model loaded.')
   
    # define loss
    criterion = nn.BCELoss() 

    # define optimizers
    G_optimizer = optim.Adam(G.parameters(), lr = args.lr)
    D_optimizer = optim.Adam(D.parameters(), lr = args.lr)

    print('Start Training :')
    
    n_epoch = args.epochs
    for epoch in trange(1, n_epoch+1, leave=True):           
        for batch_idx, (x, _) in enumerate(train_loader):
            x = x.view(-1, mnist_dim)
            if args.model == 'GAN':
                D_train(x, G, D, D_optimizer, criterion)
                G_train(x, G, D, G_optimizer, criterion)
            elif args.model == 'SAN':
                SAN_D_train(x, G, D, D_optimizer, criterion)
                SAN_G_train(x, G, D, G_optimizer, criterion)

        if epoch % 10 == 0:
            save_models(G, D, "checkpoints", args)

        if epoch % 50 == 0:
            visualize_san(
            G=G,
            D=D,
            real_loader=train_loader,
            epoch=epoch + 1,
            save=True, show=False
        )
                
    print('Training done')

        
