from scipy.ndimage import grey_erosion, grey_dilation, grey_closing, grey_opening
from parameters import *
from torch.nn import functional as f
from typing import Union


def erode(image: np.ndarray, strel: np.ndarray, origin: tuple = (0, 0)):
    assert len(image.shape) == 2
    assert len(strel.shape) == 2
    assert len(origin)
    assert origin[0] == origin[1]
    assert 0 <= origin[0] < strel.shape[0]
    origin = - strel.shape[0] // 2 + 1 + origin[0]
    return grey_erosion(image, origin=origin, structure=strel, mode='constant')


def dilate(image, strel, origin):
    assert len(image.shape) == 2
    assert len(strel.shape) == 2
    assert len(origin)
    assert origin[0] == origin[1]
    assert 0 <= origin[0] < strel.shape[0]
    origin = - strel.shape[0] // 2 + 1 + origin[0]
    return grey_dilation(image, origin=origin, structure=strel, mode='constant')


def closing(image, strel, origin=None):
    # TODO: change origin from relative to absolute
    if not origin:
        return grey_closing(image, structure=strel, mode='constant')
    else:
        return grey_closing(image, origin=origin, structure=strel, mode='constant')


def opening(image, strel, origin=None):
    # TODO: change origin from relative to absolute
    if not origin:
        return grey_opening(image, structure=strel, mode='constant')
    else:
        return grey_opening(image, origin=origin, structure=strel, mode='constant')


def erosion(image: torch.Tensor, strel: torch.Tensor, origin: tuple = (0, 0), border_value: Union[int, float] = 0):
    image_pad = f.pad(image, [origin[0], strel.shape[0] - origin[0] - 1, origin[1], strel.shape[1] - origin[1] - 1],
                      mode='constant', value=border_value)
    image_extended = f.unfold(image_pad.unsqueeze(0).unsqueeze(0), kernel_size=strel.shape)
    strel_flatten = torch.flatten(strel).unsqueeze(0).unsqueeze(-1)
    differences = image_extended - strel_flatten
    result, _ = differences.min(dim=1)
    return torch.reshape(result, image.shape)


def erosion_fast(image: torch.Tensor, strel: torch.Tensor, origin: tuple = (0, 0), border_value: Union[int, float] = 0):
    result, _ = (f.unfold(f.pad(image, [origin[0], strel.shape[0] - origin[0] - 1, origin[1], strel.shape[1] - origin[1] - 1], mode='constant',
                                value=border_value).unsqueeze(0).unsqueeze(0), kernel_size=strel.shape)
                 - torch.flatten(strel).unsqueeze(0).unsqueeze(-1)).min(dim=1)
    return torch.reshape(result, image.shape)


def dilation(image: torch.Tensor, strel: torch.Tensor, origin: tuple = (0, 0), border_value: Union[int, float] = 0):
    image_pad = f.pad(image, [origin[0], strel.shape[0] - origin[0] - 1, origin[1], strel.shape[1] - origin[1] - 1],
                      mode='constant', value=border_value)
    image_extended = f.unfold(image_pad.unsqueeze(0).unsqueeze(0), kernel_size=strel.shape)
    strel_flatten = torch.flatten(strel).unsqueeze(0).unsqueeze(-1)
    sums = image_extended + strel_flatten
    result, _ = sums.max(dim=1)
    return torch.reshape(result, image.shape)


if __name__ == '__main__':
    library = 'cv2'
    operation = 'dilation'

    _image = np.zeros((5, 5))
    _image[1, :] = 1
    _image[:, 1] = 2
    _image[3, :] = 3
    _image[:, 3] = 3

    plt.figure()
    plt.imshow(_image, cmap='Greys', vmin=_image.min(), vmax=_image.max(), origin='lower')

    _strel = np.zeros((3, 3)) - 1000
    _strel[1, :] = 0
    _strel[:, 1] = 0

    _origin = (1, 1)

    plt.figure()
    plt.imshow(_strel, cmap='Greys', vmin=-100, vmax=0, origin='lower')
    plt.scatter(_origin[0], _origin[1], marker='x', c='r')

    if library == 'scipy':
        if operation == 'erosion':
            _erosion = erode(_image, _strel, origin=_origin)
            plt.figure()
            plt.imshow(_erosion, cmap='Greys', vmin=_image.min(), vmax=_image.max(), origin='lower')

        elif operation == 'dilation':
            _dilation = dilate(_image, _strel, origin=_origin)
            plt.figure()
            plt.imshow(_dilation, cmap='Greys', vmin=_image.min(), vmax=_image.max(), origin='lower')

    if library == 'cv2':
        from cv2 import dilate as dilate_cv2
        from cv2 import erode as erode_cv2
        from cv2 import getStructuringElement

        if operation == 'erosion':
            _erosion = erode_cv2(_image.astype(np.int8), np.exp(_strel).astype(np.int8))
            plt.figure()
            plt.imshow(_erosion, cmap='Greys', vmin=_image.min(), vmax=_image.max(), origin='lower')

        elif operation == 'dilation':
            k_size = _strel.shape[0]
            _dilation = dilate_cv2(_image.astype(np.int8), getStructuringElement(0, k_size))
            plt.figure()
            plt.imshow(_dilation, cmap='Greys', vmin=_image.min(), vmax=_image.max(), origin='lower')

    elif library == 'torch':
        _image_tensor = torch.tensor(_image, device=DEVICE)
        _strel_tensor = torch.tensor(_strel, device=DEVICE)

        if operation == 'erosion':
            _erosion_tensor = erosion(_image_tensor, _strel_tensor, origin=_origin)
            plt.figure()
            plt.imshow(_erosion_tensor.cpu().numpy(), cmap='Greys', vmin=_image.min(), vmax=_image.max(), origin='lower')
        elif operation == 'dilation':
            _dilation_tensor = dilation(_image_tensor, _strel_tensor, origin=_origin, border_value=-1000)
            plt.figure()
            plt.imshow(_dilation_tensor.cpu().numpy(), cmap='Greys', vmin=_image.min(), vmax=_image.max(), origin='lower')

    elif library == 'kornia':
        from kornia.morphology import erosion as erosion_kornia
        from kornia.morphology import dilation as dilation_kornia

        _image_tensor = torch.tensor(_image, device=DEVICE)
        _strel_tensor = torch.tensor(_strel, device=DEVICE)

        if operation == 'erosion':
            _erosion_tensor = erosion_kornia(_image_tensor.unsqueeze(0).unsqueeze(0), _strel_tensor)
            plt.figure()
            plt.imshow(_erosion_tensor.cpu().numpy()[0, 0, :, :], cmap='Greys', vmin=_image.min(), vmax=_image.max(), origin='lower')
        elif operation == 'dilation':
            _dilation_tensor = dilation_kornia(_image_tensor.unsqueeze(0).unsqueeze(0), _strel_tensor)
            plt.figure()
            plt.imshow(_dilation_tensor.cpu().numpy()[0, 0, :, :], cmap='Greys', vmin=_image.min(), vmax=_image.max(), origin='lower')
