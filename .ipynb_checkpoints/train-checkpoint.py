import os
import argparse

import torch
import torch.nn as nn
import torchvision
import torch.optim as optim

from dataset import *
from model.Discriminator_model import Discriminator
from model.Generator_model import Generator
from utils.save_img import save_tensor_images
from utils.noise import get_noise

from tqdm.auto import tqdm


def get_args():
    ''' Get Aruments from terminal
    
    batch_size: batch size used in dataloader
    lr: learning rate for training
    epochs: total epochs for trainging
    save_dir: the directory name to save generated[fake] images
    noise_dim: the dimension of noise vector[Z]
    '''
    parser = argparse.ArgumentParser(description='Set the hyperparameters of DCGANs',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('--batch_size', '-b', default=128,
                       help='The Batch Size used in dataset')
    parser.add_argument('--lr', '-l', default=0.0002,
                       help='Learning rate of DCGANs. recommend 0.0002')
    parser.add_argument('--epochs', '-e', default=100,
                       help='The number of epochs to train')
    parser.add_argument('--save_dir', '-d', default='results',
                       help='The images directory where to save. Do not use [data], [utils] for directory name.')
    parser.add_argument('--save_step', '-s', default=500,
                       help='Interval of saving image')
    parser.add_argument('--noise_dim', '-z', default=64,
                       help='Dimension of noise vector Z')
    
    return parser.parse_args()

def show_hyperparameters(args):
    ''' print hyperparameters on screen '''
    
    print( '       +================================================+')
    print( '       |               Hyperparameters                  |')
    print( '       +================================================+')
    print(f'       | dimension of noise vector: {args.noise_dim}')
    print(f'       | batch_size:                {args.batch_size}')
    print(f'       | learning rate:             {args.lr}')
    print(f'       | epochs:                    {args.epochs}')
    print(f'       | directory name to save:    {args.save_dir}')
    print(f'       | saving steps:              {args.save_step}')
    print( '       +================================================+')
    
    return

def save_models(G, D):
    '''Save Generator, Discriminator
    Parameters:
        G: Generator model
        D: Discriminator model
    '''
    print('Save the Generator, Discriminator')
    print('Please wait...')
    torch.save(G.state_dict(), 'DCGAN_G.pt')
    torch.save(D.state_dict(), 'DCGAN_D.pt')
    print('Saving the models is Completed')
    
    return



def weights_init(m):
    ''' Initialize the weights of the model
    Parameters
        m: model
    '''

    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    if isinstance(m, nn.BatchNorm2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
        torch.nn.init.constant_(m.bias, 0)
    return


def train(G, D,
          noise_dim,
          batch_size,
          lr,
          epochs,
          save_dir,
          save_step):
    ''' Train the G, D
    Parameters:
        G: Generator
        D: Discriminator
        batch_size: batch size
        lr: learning rate
        epochs: total epochs to train
        save_dir: the directory name to save generated[fake] images
        save_step: the step to save the images
    '''
    
    # D only discriminate real/fake, so use Binary Cross Entropy loss function
    criterion = nn.BCEWithLogitsLoss()
    
    # make directory [save_dir]
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f'Create the directory [{save_dir}]')
    
    beta_1 = 0.5 
    beta_2 = 0.999
    
    # setting optimizer
    G_optim = torch.optim.Adam(G.parameters(), lr=lr, betas=(beta_1, beta_2))
    D_optim = torch.optim.Adam(D.parameters(), lr=lr, betas=(beta_1, beta_2))
    
    # initialize the weights
    G = G.apply(weights_init)
    D = D.apply(weights_init)
    
    # get DataLoaer
    dataloader = get_data_loader(batch_size=batch_size)
    
    step = 1
    img_id = 1
    
    mean_G_loss = 0
    mean_D_loss = 0
    
    for epoch in range(epochs):
        # Dataloader returns the batches
        for real, _ in tqdm(dataloader):
            cur_batch_size = len(real)
            real = real.to(device)

            # ========== Update discriminator ==========
            fake_noise = get_noise(cur_batch_size, noise_dim, device=device)
            D_optim.zero_grad()
            fake = G(fake_noise)
            
            D_fake_pred = D(fake.detach())
            D_fake_loss = criterion(D_fake_pred, torch.zeros_like(D_fake_pred))
            
            D_real_pred = D(real)
            D_real_loss = criterion(D_real_pred, torch.ones_like(D_real_pred))
            D_loss = (D_fake_loss + D_real_loss) / 2 # average of 2 losses

            # Keep track of the average discriminator loss
            mean_D_loss += D_loss.item()
            D_loss.backward(retain_graph=True)
            D_optim.step()

            
            ## ========== Update generator ========== ##
            fake_noise_2 = get_noise(cur_batch_size, noise_dim, device=device)
            G_optim.zero_grad()
            fake_2 = G(fake_noise_2)
            
            D_fake_pred = D(fake_2)
            G_loss = criterion(D_fake_pred, torch.ones_like(D_fake_pred))
            G_loss.backward()
            G_optim.step()

            # Keep track of the average generator loss
            mean_G_loss += G_loss.item()
            
            ## Save the images for every [save_step] iterations ##
            if step % save_step == 0:
                fake_noise = get_noise(cur_batch_size, noise_dim, device=device)
                fake = G(fake_noise)
                
                #real_image_path = os.path.join(save_dir, 'real_images_{}.png'.format(img_id))
                fake_image_path = os.path.join(save_dir, 'fake_images_{}.png'.format(img_id))

                save_tensor_images(fake, img_path=fake_image_path)
                #save_tensor_images(real, img_path=real_image_path)
                img_id += 1
                
            step += 1
        
        # show losses per epoch
        mean_G_loss /= len(dataloader)
        mean_D_loss /= len(dataloader)
        print(f"Generator loss: {mean_G_loss}, discriminator loss: {mean_D_loss}")
        mean_G_loss = mean_D_loss = 0
    
    print(f'Create {img_id}s fake images')
    save_models(G, D)
    
    return # end train()


if __name__ == '__main__':
    args = get_args()
    show_hyperparameters(args)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'You are using [{device}]')
    
    G = Generator(args.noise_dim).to(device)
    D = Discriminator().to(device) 
    
    try:
        train(G, D,
              noise_dim=args.noise_dim,
              batch_size=args.batch_size,
              lr=args.lr,
              epochs=args.epochs,
              save_dir=args.save_dir,
              save_step=args.save_step)
        
    except KeyboardInterrupt:
        save_models(G, D)