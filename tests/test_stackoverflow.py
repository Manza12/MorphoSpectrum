import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import grey_dilation as dilation_scipy
from kornia.morphology import dilation as dilation_kornia
import torch
from torch.nn import functional as f


def dilation_pytorch(image, strel, origin=(0, 0), border_value=0):
    image_pad = f.pad(image, [origin[0], strel.shape[0] - origin[0] - 1, origin[1], strel.shape[1] - origin[1] - 1],
                      mode='constant', value=border_value)
    image_extended = f.unfold(image_pad.unsqueeze(0).unsqueeze(0), kernel_size=strel.shape)
    strel_flatten = torch.flatten(strel).unsqueeze(0).unsqueeze(-1)
    sums = image_extended + strel_flatten
    result, _ = sums.max(dim=1)
    return torch.reshape(result, image.shape)


image = np.zeros((5, 5))
image[1, :] = 1
image[:, 1] = 2
image[3, :] = 3
image[:, 3] = 3

plt.figure()
plt.imshow(image, cmap='Greys', vmin=image.min(), vmax=image.max(), origin='lower')
plt.title('Original image')

strel = np.zeros((3, 3)) - 1000
strel[0, :] = 0
strel[:, 0] = 0

origin = (0, 0)

plt.figure()
plt.imshow(strel, cmap='Greys', vmin=-100, vmax=0, origin='lower')
plt.scatter(origin[0], origin[1], marker='x', c='r')
plt.title('Structural element')

# Scipy
dilated_image_scipy = dilation_scipy(image, structure=strel, mode='constant', origin=0)

plt.figure()
plt.imshow(dilated_image_scipy, cmap='Greys', vmin=image.min(), vmax=image.max(), origin='lower')
plt.title('Dilated image - Scipy')

# PyTorch
image_tensor = torch.tensor(image)
strel_tensor = torch.tensor(strel)
dilated_image_pytorch = dilation_pytorch(image_tensor, strel_tensor, origin=origin, border_value=-1000)

plt.figure()
plt.imshow(dilated_image_pytorch.cpu().numpy(), cmap='Greys', vmin=image.min(), vmax=image.max(), origin='lower')
plt.title('Dilated image - PyTorch')

# Kornia
dilated_image_kornia = dilation_kornia(image_tensor.unsqueeze(0).unsqueeze(0), strel_tensor)
plt.figure()
plt.imshow(dilated_image_kornia.cpu().numpy()[0, 0, :, :], cmap='Greys', vmin=image.min(), vmax=image.max(), origin='lower')
plt.title('Dilated image - Kornia')

plt.show()
