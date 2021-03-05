import time
import torch
import numpy as np
from scipy.ndimage import grey_erosion as erosion_scipy
from mathematical_morphology import erosion as erosion_torch
from mathematical_morphology import erosion_fast

# Parameters
loops = 10
N = 1000
M = 10

# Image
size_image = (N, N)
image = np.random.random(size_image)

# Strel
size_strel = (M, M)
strel = np.random.random(size_strel)

# Scipy
sta = time.time()
for i in range(loops):
    eroded_image_scipy = erosion_scipy(image, structure=strel, mode='constant', origin=0)
end = time.time()
time_scipy = end-sta
print('Time to erode Scipy:       ', round(time_scipy, 6), 'seconds', '(reference)')

# PyTorch
image_tensor_cuda = torch.tensor(image, device='cuda:0')
strel_tensor_cuda = torch.tensor(strel, device='cuda:0')

# PyTorch CUDA
sta = time.time()
for i in range(loops):
    eroded_image_torch = erosion_torch(image_tensor_cuda, strel_tensor_cuda)
end = time.time()
time_cuda = end-sta
print('Time to erode PyTorch CUDA:', round(time_cuda, 6), 'seconds', '(x%r)' % (round(time_scipy / time_cuda)))

# PyTorch CUDA - Fast
sta = time.time()
for i in range(loops):
    eroded_image_fast = erosion_fast(image_tensor_cuda, strel_tensor_cuda)
end = time.time()
time_fast = end-sta
print('Time to erode PyTorch Fast:', round(time_fast, 6), 'seconds', '(x%r)' % (round(time_scipy / time_fast)))

# PyTorch CUDA
sta = time.time()
for i in range(loops):
    eroded_image_torch = erosion_torch(image_tensor_cuda, strel_tensor_cuda)
end = time.time()
time_cuda = end-sta
print('Time to erode PyTorch CUDA:', round(time_cuda, 6), 'seconds', '(x%r)' % (round(time_scipy / time_cuda)))

# PyTorch CPU
image_tensor_cpu = torch.tensor(image)
strel_tensor_cpu = torch.tensor(strel)

sta = time.time()
for i in range(loops):
    eroded_image_torch_cpu = erosion_torch(image_tensor_cpu, strel_tensor_cpu)
end = time.time()
time_cpu = end-sta
print('Time to erode PyTorch CPU: ', round(time_cpu, 6), 'seconds', '(/%r)' % (round(time_cpu / time_scipy)))
