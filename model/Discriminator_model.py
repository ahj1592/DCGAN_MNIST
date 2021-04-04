import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, im_chan=1, hidden_dim=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(im_chan, hidden_dim, kernel_size=4, stride=2),
            nn.BatchNorm2d(hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(hidden_dim, hidden_dim*2, kernel_size=4, stride=2), 
            nn.BatchNorm2d(hidden_dim*2),
            nn.LeakyReLU(0.2, inplace=True),  
            
            nn.Conv2d(hidden_dim*2, 1, kernel_size=4, stride=2),
            
        )
    def forward(self, image):
        pred = self.net(image)
        return pred.view(len(pred), -1)