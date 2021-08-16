import torch
print('CUDA is available? ', torch.cuda.is_available())
print('CUDA device name =', torch.cuda.get_device_name(0))
print('PyTorch version = ' torch.__version__)
