import torch
print("torch:", torch.__version__)
print("cuda runtime (torch):", torch.version.cuda)
print("cuDNN version:", torch.backends.cudnn.version())