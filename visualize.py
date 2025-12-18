from model import SAN_Discriminator, Generator, GAN_Discriminator
from utils import load_model, load_model_D
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def extract_features(G, D, real_loader, z_dim=100, n_samples=2000):
    real_feats = []
    fake_feats = []

    G.eval()
    D.eval()

    with torch.no_grad():
        # Real
        for x, _ in real_loader:
            x = x.view(x.size(0), -1).to(next(D.parameters()).device)
            _, h = D(x, flg_train=False)
            real_feats.append(h)
            if len(real_feats) * x.size(0) >= n_samples:
                break

        # Fake
        z = torch.randn(n_samples, z_dim).to(next(G.parameters()).device)
        x_fake = G(z)
        _, h_fake = D(x_fake, flg_train=False)
        fake_feats.append(h_fake)

    real_feats = torch.cat(real_feats)[:n_samples].cpu().numpy() #on va concaténer toutes les features réelles
    fake_feats = torch.cat(fake_feats)[:n_samples].cpu().numpy()

    return real_feats, fake_feats


def plot_cdf(data, label):
    sorted_data = np.sort(data) #on va trier les données
    y = np.linspace(0, 1, len(sorted_data)) #on crée des valeurs y entre 0 et 1
    plt.plot(sorted_data, y, label=label) #on trace la courbe


def visualize(G, D, real_loader, epoch,
                  z_dim=100, n_samples=2000,
                  save=True, show=False):

    real_feats, fake_feats = extract_features(
        G, D, real_loader, z_dim, n_samples
    )

    # Direction SAN
    mean_diff = real_feats.mean(axis=0) - fake_feats.mean(axis=0)
    omega = mean_diff / np.linalg.norm(mean_diff) #direction normalisée

    proj_real = real_feats @ omega #projection des features sur la direction
    proj_fake = fake_feats @ omega

    plt.figure(figsize=(6,4))
    plot_cdf(proj_real, "Real")
    plot_cdf(proj_fake, "Fake")
    plt.legend()
    plt.title(f"CDF of SAN – Epoch {epoch}")

    if save:
        plt.savefig(f"san_cdf_epoch_{epoch}.png")

    if show:
        plt.show()

    plt.close()


