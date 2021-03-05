import numpy as np
import torch
from torch.nn import functional as f
from scipy.ndimage import grey_dilation as dilation_scipy
from kornia.morphology import dilation as dilation_kornia
import matplotlib.pyplot as plt


# Definition of the dilation using PyTorch
def dilation_pytorch(image, strel, origin=(0, 0), border_value=0):
    # first pad the image to have correct unfolding; here is where the origins is used
    image_pad = f.pad(image, [origin[0], strel.shape[0] - origin[0] - 1, origin[1], strel.shape[1] - origin[1] - 1], mode='constant', value=border_value)
    # Unfold the image to be able to perform operation on neighborhoods
    image_unfold = f.unfold(image_pad.unsqueeze(0).unsqueeze(0), kernel_size=strel.shape)
    # Flatten the structural element since its two dimensions have been flatten when unfolding
    strel_flatten = torch.flatten(strel).unsqueeze(0).unsqueeze(-1)
    # Perform the greyscale operation; sum would be replaced by rest if you want erosion
    sums = image_unfold + strel_flatten
    # Take maximum over the neighborhood
    result, _ = sums.max(dim=1)
    # Reshape the image to recover initial shape
    return torch.reshape(result, image.shape)


# Test image
image = np.zeros((7, 7), dtype=int)
image[2:5, 2:5] = 1
image[4, 4] = 2
image[2, 3] = 3

plt.figure()
plt.imshow(image, cmap='Greys', vmin=image.min(), vmax=image.max(), origin='lower')
plt.title('Original image')

# Structural element square 3x3
strel = np.ones((3, 3))

# Origin of the structural element
origin = (1, 1)

# Scipy
dilated_image_scipy = dilation_scipy(image, size=(3, 3), structure=strel)

plt.figure()
plt.imshow(dilated_image_scipy, cmap='Greys', vmin=image.min(), vmax=image.max(), origin='lower')
plt.title('Dilated image - Scipy')

# PyTorch
image_tensor = torch.tensor(image, dtype=torch.float)
strel_tensor = torch.tensor(strel, dtype=torch.float)
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
