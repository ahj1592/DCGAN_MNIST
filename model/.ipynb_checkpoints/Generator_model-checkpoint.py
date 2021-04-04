import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, z_dim=10, im_chan=1, hidden_dim=64):
        super().__init__()
        self.z_dim = z_dim
        
        self.net = nn.Sequential(
            nn.ConvTranspose2d(z_dim, hidden_dim*4, kernel_size=3, stride=2), 
            nn.BatchNorm2d(hidden_dim*4),
            nn.ReLU(),
            
            nn.ConvTranspose2d(hidden_dim*4, hidden_dim*2, kernel_size=4, stride=1), 
            nn.BatchNorm2d(hidden_dim*2),
            nn.ReLU(),
            
            nn.ConvTranspose2d(hidden_dim*2, hidden_dim, kernel_size=3, stride=2), 
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            
            nn.ConvTranspose2d(hidden_dim, im_chan, kernel_size=4, stride=2),
            nn.Tanh()
        )
        
        
    def forward(self, noise):
        noise = noise.view(len(noise), self.z_dim, 1, 1)
        return self.net(noise)