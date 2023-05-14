import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.datasets import ImageFolder
from WGAN_GP import WGANGP
from model import Generator, Discriminator
from utils import *
import os


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Current Device: ", device)

    # Set args
    batch_size = 128
    max_epoch = 80
    lr = 1e-4
    latent_dim = 100
    trial = 1
    current_epoch = 0
    checkpoint_dir = f'./checkpoint/cp_{trial}_{current_epoch}.pt'
    result_dir = f'./result/cp_{trial}'
    data_root = '../../Dataset/CelebB'

    if not os.path.exists(result_dir):
        os.mkdir(result_dir)

    # Define Generator and Discriminator
    gen = Generator(latent_dim, 3).to(device)
    disc = Discriminator(3).to(device)

    if os.path.exists(checkpoint_dir):
        state_dict = torch.load(checkpoint_dir)
        gen.load_state_dict(state_dict['gen'])
        disc.load_state_dict(state_dict['disc'])
        print("Checkpoint Loaded")
    else:
        gen.apply(weights_init)
        disc.apply(weights_init)
        print("New Model")

    
    # Define Transform
    tf = transforms.Compose([
        transforms.Resize((64)),
        transforms.CenterCrop(64),
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5)
    ])


    # Define Dataloder
    dataset = ImageFolder(root=data_root, transform=tf)
    dataloader = DataLoader(dataset=dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            drop_last=True,
                            num_workers=4)


    gan_trainer = WGANGP(gen=gen,
                      disc=disc,
                      max_epoch=max_epoch,
                      batch_size=batch_size,
                      lr=lr,
                      dataloader=dataloader,
                      trial=trial,
                      current_epoch=current_epoch,
                      latent_dim=latent_dim,
                      device=device
                      )
    
    gan_trainer.train()



if __name__=='__main__':
    main()


