import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor


# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


def train(dataloader, model, loss_fn, optimizer, device) -> (float, int, int):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 300 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            yield loss, current, size


def test(dataloader, model, loss_fn, device) -> (float, float):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size

    return correct, test_loss


def pick_device(device_name: str | None) -> torch.device:
    if device_name is None:
        device: torch.device = torch.device('cpu')
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif torch.xpu.is_available():
            import intel_extension_for_pytorch as ipex

            device = torch.device('xpu')
        else:
            raise ValueError('No device available!')
    else:
        device = torch.device(device_name)

    return device


def go(epochs: int = 5, batch_size: int = 64, device_name: str | None = None):
    device = pick_device(device_name)
    torch.set_default_device(device)
    print(f'device: {torch.get_default_device()}')

    # setup training data from open datasets (downloads if necessary)
    training_data = datasets.FashionMNIST(
        root='quickstart_data',
        train=True,
        download=True,
        transform=ToTensor(),
    )

    # Download test data from open datasets.
    test_data = datasets.FashionMNIST(
        root='quickstart_data',
        train=False,
        download=True,
        transform=ToTensor(),
    )

    # Create data loaders.
    train_dataloader = DataLoader(training_data, batch_size=batch_size)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)

    for t in [('training', train_dataloader), ('testing', test_dataloader)]:
        for X, y in t[1]:
            print(f'{t[0]} data: {X.shape}({X.dtype}) / {y.shape}({y.dtype})')
            break  # we only need one from each tensor

    # acc_device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else 'cpu'
    # print(f'accelerator device: {acc_device}')

    model = NeuralNetwork().to(device)
    print(f'model: {model}')

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    for t in range(epochs):
        print(f'Epoch {t + 1}\n-------------------------------')
        print(f'training:')
        for i in train(train_dataloader, model, loss_fn, optimizer, device):
            print(f'  loss: {i[0]:>7f}  [{i[1]:>5d}/{i[2]:>5d}]')
        results = test(test_dataloader, model, loss_fn, device)
        print(f'on test data: acc: {(100 * results[0]):>0.1f}%, avg-loss: {results[1]:>3f} \n')

    # try some predictions
    classes = [
        "T-shirt/top",
        "Trouser",
        "Pullover",
        "Dress",
        "Coat",
        "Sandal",
        "Shirt",
        "Sneaker",
        "Bag",
        "Ankle boot",
    ]

    model.eval()
    i = 0
    for x, y in test_data:
        with torch.no_grad():
            x = x.to(device)
            pred = model(x)
            predicted, actual = classes[pred[0].argmax(0)], classes[y]
            print(f'prediction: {predicted} ({actual})')
            i += 1
            if i > 5:
                break

    print(f'\ndevice: {torch.get_default_device()}')


go(epochs=2)
print('done')
