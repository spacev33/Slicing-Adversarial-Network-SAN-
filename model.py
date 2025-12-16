import torch
import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):
    def __init__(self, g_output_dim):
        super(Generator, self).__init__()       
        self.fc1 = nn.Linear(100, 256)
        self.fc2 = nn.Linear(self.fc1.out_features, self.fc1.out_features*2)
        self.fc3 = nn.Linear(self.fc2.out_features, self.fc2.out_features*2)
        self.fc4 = nn.Linear(self.fc3.out_features, g_output_dim)

    # forward method
    def forward(self, x): 
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = F.leaky_relu(self.fc3(x), 0.2)
        return torch.tanh(self.fc4(x))

class GAN_Discriminator(nn.Module):
    def __init__(self, d_input_dim):
        super(GAN_Discriminator, self).__init__()
        self.fc1 = nn.Linear(d_input_dim, 1024)
        self.fc2 = nn.Linear(self.fc1.out_features, self.fc1.out_features//2)
        self.fc3 = nn.Linear(self.fc2.out_features, self.fc2.out_features//2)
        self.fc4 = nn.Linear(self.fc3.out_features, 1)

    # forward method
    def forward(self, x):
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = F.leaky_relu(self.fc3(x), 0.2)
        return torch.sigmoid(self.fc4(x))


class SAN_Discriminator(nn.Module):
    def __init__(self,d_input,dim=1024, num_class=1):
        super(SAN_Discriminator, self).__init__()
        self.fc1 = nn.Linear(d_input, dim)
        self.fc2 = nn.Linear(self.fc1.out_features, self.fc1.out_features//2)
        self.fc3 = nn.Linear(self.fc2.out_features, self.fc2.out_features//2)
        self.fc_w = nn.Parameter(torch.randn(1, dim//4))
    

    
    def forward(self,x,class_ids=None, flg_train=True):
        x = F.leaky_relu(self.fc1(x), 0.1)
        x = F.leaky_relu(self.fc2(x), 0.1)
        x = F.leaky_relu(self.fc3(x), 0.1)

        h_feature = x
        h_feature = torch.flatten(h_feature, start_dim=1)

        # Recuperation des distance de la dernière couche
        omega = self.fc_w
        # On va normaliser la dernière couche
        direction = F.normalize(omega, dim=1)
        # On va calculer la norme
        norme = torch.norm(omega, dim=1).unsqueeze(1)
        # On projette
        h_feature = h_feature * norme
    
        # Séparation entre fonction et direction
        if flg_train:
            out_fun = (h_feature * direction.detach()).sum(dim=1)
            out_dir = (h_feature.detach() * direction).sum(dim=1)
            out = dict(fun=out_fun, dir=out_dir)
        else:
            out = (h_feature * direction).sum(dim=1)
        return out
