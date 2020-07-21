import argparse


def get_args():
    # Training settings.
    parser = argparse.ArgumentParser(description="PyTorch Tutorial")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        metavar="N",
        help="input batch size for training (default: 4)",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=10,
        metavar="N",
        help="input batch size for testing (default: 4)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        metavar="N",
        help="number of epochs to train (default: 2)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        metavar="LR",
        help="learning rate (default: 0.001)",
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.9,
        metavar="M",
        help="momentum (default: 0.9)",
    )
    parser.add_argument(
        "--no-cuda", action="store_true", default=False, help="disables CUDA training"
    )
    parser.add_argument(
        "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        metavar="NW",
        help="random seed (default: 1)",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        metavar="N",
        help="how many batches to wait before logging training status",
    )
    parser.add_argument(
        "--save-model",
        action="store_true",
        default=False,
        help="For Saving the current Model",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="finetuning_convnet",
        help="For setting a training mode",
    )
    parser.add_argument(
        "--check-images",
        action="store_true",
        default=False,
        help="For checking input images from dataloader",
    )
    return parser.parse_args()
