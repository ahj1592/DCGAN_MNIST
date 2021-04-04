import matplotlib.pyplot as plt
from torchvision.utils import make_grid, save_image

def save_tensor_images(image_tensor, num_images=25, size=(1, 28, 28), img_path='img.png'):
    image_tensor = (image_tensor + 1) / 2
    image_unflat = image_tensor.detach().cpu().view(-1, *size)
    image_grid = make_grid(image_unflat[:num_images], nrow=5)
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    plt.savefig(img_path)
    return