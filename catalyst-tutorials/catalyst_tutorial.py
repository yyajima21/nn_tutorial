"""
reference: Catalyst 101 — Accelerated PyTorch
link: https://medium.com/pytorch/catalyst-101-accelerated-pytorch-bd766a556d92
"""
import torch
from torch import nn

from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

from catalyst import dl
from catalyst.utils import metrics
from catalyst.contrib.nn import Flatten
from catalyst.contrib.models import SequentialNet

def main():
    # =============================================================================
    # Load and split dataset
    # =============================================================================
    
    train_dataset = MNIST("./mnist", train=True, download=True, transform=ToTensor())
    valid_dataset = MNIST("./mnist", train=False, download=True, transform=ToTensor())

    train_loader = DataLoader(train_dataset, batch_size=32)
    valid_loader = DataLoader(valid_dataset, batch_size=32)

    # =============================================================================
    # catalyst 3-layer fully connected model
    # =============================================================================
    
    model = nn.Sequential(
        Flatten(),
        SequentialNet(
            hiddens=[28 * 28, 128, 128, 10], 
            layer_fn=nn.Linear, 
            activation_fn=nn.ReLU
        )
    )

    # =============================================================================
    # Optimizer
    # =============================================================================
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # =============================================================================
    # multi-loss criterion example
    # =============================================================================
    
    criterion = nn.CrossEntropyLoss()

    # =============================================================================
    # runner object class and training code
    # =============================================================================
    
    runner = dl.SupervisedRunner()

    runner.train(
        loaders={"train": train_loader, "valid": valid_loader},
        model=model, criterion=criterion, optimizer=optimizer,
        num_epochs=1, logdir="./logs", verbose=True,
        callbacks=[dl.AccuracyCallback(num_classes=10)],
        load_best_on_end=True,
    )

    # =============================================================================
    # model inference
    # =============================================================================
    
    for prediction in runner.predict_loader(loader=valid_loader):
        assert prediction.detach().cpu().numpy().shape[-1] == 10
    
    # =============================================================================
    # model tracing
    # =============================================================================
    
    traced_model = runner.trace(loader=valid_loader)


if __name__ == "__main__":
    main()