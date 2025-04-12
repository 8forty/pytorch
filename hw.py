import torch

device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.xpu.is_available():
    device = torch.device('xpu')

torch.set_default_device(device)
print(f'device: {torch.get_default_device()}')
