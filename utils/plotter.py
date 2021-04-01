import sys
import os
import argparse

PACKAGE_PARENT = ".."
SCRIPT_DIR = os.path.dirname(
    os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__)))
)
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

from utils.memory_management import load_obj

import matplotlib.pyplot as plt

def plot_active(init=100, step=100):
    train_losses = load_obj("train_loss_list")
    train_f1s = load_obj("train_f1_list")

    val_losses = load_obj("val_loss_list")
    val_f1s = load_obj("val_f1_list")

    total_epochs = len(train_losses) * step + init

    plt.figure(5)

    plt.ylabel("Loss")
    plt.title("Loss during active learning")
    plt.xlabel("Dataset len")

    plt.plot(
        range(init, total_epochs, step), train_losses,
        color="red",
        label="train set"
    )

    plt.plot(
        range(init, total_epochs, step), val_losses,
        color="blue",
        label="valid set"
    )

    plt.legend(loc="upper left")

    plt.tight_layout()

    plt.savefig("plots/active_loss_i" + str(init) + "_s" + str(step) + ".png")

    plt.figure(6)

    plt.ylabel("F1")
    plt.title("F1 during active learning")
    plt.xlabel("Dataset len")

    plt.plot(
        range(init, total_epochs, step), train_f1s,
        color="red",
        label="train set"
    )

    plt.plot(
        range(init, total_epochs, step), val_f1s,
        color="blue",
        label="valid set"
    )

    plt.legend(loc="upper left")

    plt.tight_layout()

    plt.savefig("plots/active_f1_i" + str(init) + "_s" + str(step) + ".png")

def plot_in_comparison(since_epoch):
    train_losses = load_obj("train_loss_list")[since_epoch:]
    train_f1s = load_obj("train_f1_list")[since_epoch:]

    val_losses = load_obj("val_loss_list")[since_epoch:]
    val_f1s = load_obj("val_f1_list")[since_epoch:]

    total_epochs = len(train_losses) + since_epoch

    plt.figure(3)

    plt.ylabel("Loss")
    plt.title("Loss after " + str(since_epoch) + " epochs")
    plt.xlabel("Epochs")

    plt.plot(
        range(since_epoch, total_epochs), train_losses,
        color="red",
        label="train set"
    )

    plt.plot(
        range(since_epoch, total_epochs), val_losses,
        color="blue",
        label="valid set"
    )

    plt.legend(loc="upper left")

    plt.tight_layout()

    plt.savefig("plots/loss_" + str(since_epoch) + ".png")

    plt.figure(4)

    plt.ylabel("F1")
    plt.title("F1 after " + str(since_epoch) + " epochs")
    plt.xlabel("Epochs")

    plt.plot(
        range(since_epoch, total_epochs), train_f1s,
        color="red",
        label="train set"
    )

    plt.plot(
        range(since_epoch, total_epochs), val_f1s,
        color="blue",
        label="valid set"
    )

    plt.legend(loc="upper left")

    plt.tight_layout()

    plt.savefig("plots/f1_" + str(since_epoch) + ".png")


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

def plot_mini():
    mini_losses = load_obj("mini_loss_list")
    mini_f1s = load_obj("mini_f1_list")

    plt.figure(1)

    plt.subplot(121)
    plt.ylabel("Loss")
    plt.title("Loss on mini set")
    plt.xlabel("Epochs")
    plt.plot(mini_losses)

    plt.subplot(122)
    plt.ylabel("F1")
    plt.title("F1 on mini set")
    plt.xlabel("Epochs")
    plt.plot(mini_f1s)

    plt.tight_layout()

    plt.savefig("plots/mini.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--since", type=int, default=0)

    args = parser.parse_args()

    plot_last_run()
    plot_in_comparison(args.since)