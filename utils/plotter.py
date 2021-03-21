import sys
import os

PACKAGE_PARENT = ".."
SCRIPT_DIR = os.path.dirname(
    os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__)))
)
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

from utils.memory_management import load_obj

import matplotlib.pyplot as plt

def plot_last_run():
    train_losses = load_obj("train_loss_list")
    train_f1s = load_obj("train_f1_list")

    val_losses = load_obj("val_loss_list")
    val_f1s = load_obj("val_f1_list")

    plt.figure(1)

    plt.subplot(121)
    plt.ylabel("Loss")
    plt.title("Loss on training set")
    plt.xlabel("Epochs")
    plt.plot(train_losses)

    plt.subplot(122)
    plt.ylabel("F1")
    plt.title("F1 on training set")
    plt.xlabel("Epochs")
    plt.plot(train_f1s)

    plt.tight_layout()

    plt.savefig("plots/train.png")

    plt.figure(2)
    plt.subplot(121)
    plt.ylabel("Loss")
    plt.title("Loss on validation set")
    plt.xlabel("Epochs")
    plt.plot(val_losses)

    plt.subplot(122)
    plt.ylabel("F1")
    plt.title("F1 on validation set")
    plt.xlabel("Epochs")
    plt.plot(val_f1s)

    plt.tight_layout()

    plt.savefig("plots/valid.png")

if __name__ == "__main__":
    plot_last_run()