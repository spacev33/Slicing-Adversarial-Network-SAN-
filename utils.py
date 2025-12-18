import torch
import os

device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = "mps" if torch.backends.mps.is_available() else device

#=======================GAN=======================#

def D_train(x, G, D, D_optimizer, criterion):
    #=======================Train the discriminator=======================#
    D.zero_grad()

    # train discriminator on real
    x_real, y_real = x, torch.ones(x.shape[0], 1)
    x_real, y_real = x_real.to(device), y_real.to(device)

    D_output, _ = D(x_real)
    D_real_loss = criterion(D_output, y_real)
    D_real_score = D_output

    # train discriminator on facke
    z = torch.randn(x.shape[0], 100).to(device)
    x_fake, y_fake = G(z), torch.zeros(x.shape[0], 1).to(device)

    D_output,_ =  D(x_fake)
    
    D_fake_loss = criterion(D_output, y_fake)
    D_fake_score = D_output

    # gradient backprop & optimize ONLY D's parameters
    D_loss = D_real_loss + D_fake_loss
    D_loss.backward()
    D_optimizer.step()
        
    return  D_loss.data.item()


def G_train(x, G, D, G_optimizer, criterion):
    #=======================Train the generator=======================#
    G.zero_grad()

    z = torch.randn(x.shape[0], 100).to(device)
    y = torch.ones(x.shape[0], 1).to(device)
                 
    G_output = G(z)
    D_output,_ = D(G_output)
    G_loss = criterion(D_output, y)

    # gradient backprop & optimize ONLY G's parameters
    G_loss.backward()
    G_optimizer.step()
        
    return G_loss.data.item()


#=======================SAN=======================#

def SAN_D_train(x, G, D, D_optimizer, criterion):
    #=======================Train the discriminator=======================#
    D.zero_grad()

    # train discriminator on real
    x_real, y_real = x, torch.ones(x.shape[0], 1)
    x_real, y_real = x_real.to(device), y_real.to(device)

    D_output, _ = D(x_real, flg_train=True)
    loss_real = san_loss(D_output, is_real=True)

    
    # train discriminator on fake
    z = torch.randn(x.shape[0], 100).to(device)
    
    #D_output =  D(x_fake, flg_train=True)
    x_fake = G(z).detach()
    out_fake, _ = D(x_fake, flg_train=True)
    loss_fake = san_loss(out_fake, is_real=False)
    
    # gradient backprop & optimize ONLY D's parameters
    loss_D = loss_real + loss_fake
    loss_D.backward()
    D_optimizer.step()
        
    return  loss_D.data.item()


def SAN_G_train(x, G, D, G_optimizer, criterion):
    #=======================Train the generator=======================#
    G.zero_grad()

    z = torch.randn(x.shape[0], 100).to(device)
    y = torch.ones(x.shape[0], 1).to(device)
                 
    G_output = G(z)
    D_output, _ = D(G_output, flg_train=False)
    G_loss = torch.relu(1 - D_output.view(-1, 1)).mean()

    # gradient backprop & optimize ONLY G's parameters
    G_loss.backward()
    G_optimizer.step()
        
    return G_loss.data.item()

def san_loss(disc_out, is_real):
    fun = disc_out["fun"]
    dir = disc_out["dir"]

    if is_real:
        loss_fun = torch.relu(1 - fun).mean()
        loss_dir = - dir.mean()
    else:
        loss_fun = torch.relu(1 + fun).mean()
        loss_dir = dir.mean()

    return loss_fun + loss_dir


def save_models(G, D, args):
    torch.save(G.state_dict(), os.path.join(args.model,'G.pth'))
    torch.save(D.state_dict(), os.path.join(args.model,'D.pth'))

def load_model(G, folder):
    ckpt = torch.load(os.path.join(folder,'G.pth'), map_location = "cpu")
    G.load_state_dict({k.replace('module.', ''): v for k, v in ckpt.items()})
    return G

def load_model_D(D, folder):
    ckpt = torch.load(os.path.join(folder,'D.pth'), map_location = "cpu")
    D.load_state_dict({k.replace('module.', ''): v for k, v in ckpt.items()})
    return D
