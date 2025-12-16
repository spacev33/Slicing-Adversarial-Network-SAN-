import torch
import os

device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = "mps" if torch.backends.mps.is_available() else device

def D_train(x, G, D, D_optimizer, criterion):
    n_projections=50
    #=======================Train the discriminator=======================#
    D.zero_grad()

    # ======================= Train the critic ======================= #
    D.zero_grad()

    # real samples
    x_real = x.to(device)
    h_real = D(x_real)             # (batch, d)

    # fake samples
    z = torch.randn(x.shape[0], 100, device=device)
    x_fake = G(z).detach()          # IMPORTANT
    h_fake = D(x_fake)

    # sliced Wasserstein loss (critic wants to MAXIMIZE it)
    D_loss = sliced_wasserstein_distance(h_real, h_fake, n_projections)

    D_loss.backward()
    D_optimizer.step()

    return D_loss.item()

def sliced_wasserstein_distance(X, Y, n_projections=50):
    """
    X, Y : tensors of shape (batch_size, d)
    n_projections : number of random directions
    """

    device = X.device
    d = X.shape[1]

    # 1. Sample random directions on the unit sphere


    sw = 0.0
    for _ in range(n_projections):
        theta = torch.randn(d, device=device)
        theta = theta / torch.norm(theta)

        proj_X = X @ theta  # (batch_size,)
        proj_Y = Y @ theta # (batch_size,)

        proj_X_sorted, _ = torch.sort(proj_X)
        proj_Y_sorted, _ = torch.sort(proj_Y)

        # 4. Compute distance
        sw += torch.mean(torch.abs(proj_X_sorted - proj_Y_sorted))

    return sw / n_projections

def SAN_D_train(x, G, D, D_optimizer, criterion):
    #=======================Train the discriminator=======================#
    D.zero_grad()

    # train discriminator on real
    x_real, y_real = x, torch.ones(x.shape[0], 1)
    x_real, y_real = x_real.to(device), y_real.to(device)

    D_output = D(x_real, flg_train=True)
    loss_real = san_loss(D_output, is_real=True)

    
    # train discriminator on fake
    z = torch.randn(x.shape[0], 100).to(device)
    
    #D_output =  D(x_fake, flg_train=True)
    x_fake = G(z).detach()
    out_fake = D(x_fake, flg_train=True)
    loss_fake = san_loss(out_fake, is_real=False)
    
    # gradient backprop & optimize ONLY D's parameters
    loss_D = loss_real + loss_fake
    loss_D.backward()
    D_optimizer.step()
        
    return  loss_D.data.item()

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


def SAN_G_train(x, G, D, G_optimizer, criterion):
    #=======================Train the generator=======================#
    G.zero_grad()

    z = torch.randn(x.shape[0], 100).to(device)
    y = torch.ones(x.shape[0], 1).to(device)
                 
    G_output = G(z)
    D_output = D(G_output, flg_train=False)
    G_loss = torch.relu(1 - D_output.view(-1, 1)).mean()

    # gradient backprop & optimize ONLY G's parameters
    G_loss.backward()
    G_optimizer.step()
        
    return G_loss.data.item()

def G_train(x, G, D, G_optimizer, criterion):
    n_projections=50
    #=======================Train the generator=======================#
    G.zero_grad()

    z = torch.randn(x.shape[0], 100, device=device)
    x_fake = G(z)
    h_fake = D(x_fake)

    with torch.no_grad():
        h_real = D(x.to(device))

    G_loss = sliced_wasserstein_distance(h_real, h_fake, n_projections)


    G_loss.backward()
    G_optimizer.step()

    return G_loss.item()



def save_models(G, D, folder, args):
    if args.model == 'SAN':
        torch.save(G.state_dict(), os.path.join(folder,'G.pth'))
        torch.save(D.state_dict(), os.path.join(folder,'D.pth'))
    if args.model == 'GAN':
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
