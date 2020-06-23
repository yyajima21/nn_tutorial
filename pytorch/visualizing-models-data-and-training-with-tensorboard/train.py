# imports
from __future__ import print_function
import argparse

import matplotlib.pyplot as plt
import numpy as np

import torch
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import model.model

import utils

# tensorboard
from torch.utils.tensorboard import SummaryWriter

# tensorboard (Google Colab)
import tensorflow as tf
import tensorboard as tb
tf.io.gfile = tb.compat.tensorflow_stub.io.gfile

# constant for classes
classes = ('T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
            'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot')

def parse_args():
    # Training settings
    parser = argparse.ArgumentParser(description='Tensorboard Tutorial Example')
    parser.add_argument('--train-batch-size', type=int, default=4, metavar='N',
                        help='input batch size for training (default: 4)')
    parser.add_argument('--test-batch-size', type=int, default=4, metavar='N',
                        help='input batch size for testing (default: 4)')
    parser.add_argument('--epochs', type=int, default=2, metavar='N',
                        help='number of epochs to train (default: 2)')
    parser.add_argument('--num_workers', type=int, default=2, metavar='N',
                        help='number of num workers (default: 2)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='momentum (default: 0.9)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    return parser.parse_args()

def select_n_random(data, labels, n=100):
    '''
    Selects n random datapoints and their corresponding labels from a dataset
    '''
    assert len(data) == len(labels)

    perm = torch.randperm(len(data))
    return data[perm][:n], labels[perm][:n]

def images_to_probs(net, images):
    '''
    Generates predictions and corresponding probabilities from a trained
    network and a list of images
    '''
    output = net(images)
    # convert output probabilities to predicted class
    _, preds_tensor = torch.max(output, 1)
    preds = np.squeeze(preds_tensor.numpy())
    return preds, [F.softmax(el, dim=0)[i].item() for i, el in zip(preds, output)]

def plot_classes_preds(net, images, labels):
    '''
    Generates matplotlib Figure using a trained network, along with images
    and labels from a batch, that shows the network's top prediction along
    with its probability, alongside the actual label, coloring this
    information based on whether the prediction was correct or not.
    Uses the "images_to_probs" function.
    '''
    preds, probs = images_to_probs(net, images)
    # plot the images in the batch, along with predicted and true labels
    fig = plt.figure(figsize=(12, 48))
    for idx in np.arange(4):
        ax = fig.add_subplot(1, 4, idx+1, xticks=[], yticks=[])
        matplotlib_imshow(images[idx], one_channel=True)
        ax.set_title("{0}, {1:.1f}%\n(label: {2})".format(
            classes[preds[idx]],
            probs[idx] * 100.0,
            classes[labels[idx]]),
                    color=("green" if preds[idx]==labels[idx].item() else "red"))
    return fig

def train(args, net, device, trainloader, criterion, optimizer, epoch, writer):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        if i % 1000 == 999:    # every 1000 mini-batches...
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 1000))
            # ...log the running loss
            writer.add_scalar('training loss',
                                running_loss / 1000,
                                epoch * len(trainloader) + i)
            # ...log a Matplotlib Figure showing the model's predictions on a
            # random mini-batch
            writer.add_figure('predictions vs. actuals', plot_classes_preds(net, inputs, labels), global_step=epoch * len(trainloader) + i)
            running_loss = 0.0
    print('Finished Training')
    return writer

def add_pr_curve_tensorboard(class_index, test_probs, test_preds, writer, global_step=0):
    '''
    Takes in a "class_index" from 0 to 9 and plots the corresponding
    precision-recall curve
    '''
    tensorboard_preds = test_preds == class_index
    tensorboard_probs = test_probs[:, class_index]

    writer.add_pr_curve(classes[class_index],
                        tensorboard_preds,
                        tensorboard_probs,
                        global_step=global_step)

def main():
    args = parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 2, 'pin_memory': True} if use_cuda else {}

    # transforms
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))])

    # datasets
    trainset = torchvision.datasets.FashionMNIST('./data',
        download=True,
        train=True,
        transform=transform)
    testset = torchvision.datasets.FashionMNIST('./data',
        download=True,
        train=False,
        transform=transform)

    # dataloaders
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.train_batch_size,
                                            shuffle=True, num_workers=args.num_workers)

    testloader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size,
                                            shuffle=False, num_workers=args.num_workers)
    
    net = model.model.Net()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum)

    # default `log_dir` is "runs" - we'll be more specific here
    writer = SummaryWriter('runs/fashion_mnist_experiment_1')

    # get some random training images
    dataiter = iter(trainloader)
    images, labels = dataiter.next()

    # create grid of images
    img_grid = torchvision.utils.make_grid(images)

    # show images (verify data image)
    #utils.matplotlib_imshow(img_grid, one_channel=True)
    
    # write to tensorboard
    writer.add_image('four_fashion_mnist_images', img_grid)

    writer.add_graph(net, images)

    # select random images and their target indices
    images, labels = select_n_random(trainset.data, trainset.targets)

    # get the class labels for each image
    class_labels = [classes[lab] for lab in labels]

    # log embeddings
    features = images.view(-1, 28 * 28)
    writer.add_embedding(features,
                        metadata=class_labels,
                        label_img=images.unsqueeze(1))

    for epoch in range(args.epochs):
        writer = train(args, net, device, trainloader, criterion, optimizer, epoch)

    # 1. gets the probability predictions in a test_size x num_classes Tensor
    # 2. gets the preds in a test_size Tensor
    # takes ~10 seconds to run
    class_probs = []
    class_preds = []
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            output = net(images)
            class_probs_batch = [F.softmax(el, dim=0) for el in output]
            _, class_preds_batch = torch.max(output, 1)

            class_probs.append(class_probs_batch)
            class_preds.append(class_preds_batch)

    test_probs = torch.cat([torch.stack(batch) for batch in class_probs])
    test_preds = torch.cat(class_preds)

    # plot all the pr curves
    for i in range(len(classes)):
        add_pr_curve_tensorboard(i, test_probs, test_preds, writer)
    
    writer.close()
    
if __name__ == '__main__':
    main()