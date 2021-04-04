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

#from celluloid import Camera
from collections import defaultdict

def plot_active(init=100, step=100, idd=100, suff="active_", pref="", coef=None):
    train_losses = load_obj(suff + "train_loss_list" + pref)
    train_f1s = load_obj(suff + "train_f1_list" + pref)

    val_losses = load_obj(suff + "val_loss_list" + pref)
    val_f1s = load_obj(suff + "val_f1_list" + pref)

    if coef == None:
        epoch_dynamic = list(range(init, len(train_losses) * step + init, step))
    else:
        epoch_dynamic = [init]
        for i in range(len(train_losses) - 1):
            epoch_dynamic.append(epoch_dynamic[-1] + epoch_dynamic[-1] * coef)

    plt.figure(5 + idd)

    plt.ylabel("Loss")
    plt.title("Loss during active learning")
    plt.xlabel("Dataset len")

    plt.plot(
        epoch_dynamic, train_losses,
        color="red",
        label="train set"
    )

    plt.plot(
        epoch_dynamic, val_losses,
        color="blue",
        label="valid set"
    )

    plt.legend(loc="upper left")

    plt.tight_layout()

    if coef == None:
        plt.savefig("plots/active_loss_i" + str(init) + "_s" + str(step) + pref + ".png")
    else:
        plt.savefig("plots/active_loss_i" + str(init) + "_coef.png")

    plt.figure(6 + idd)

    plt.ylabel("F1")
    plt.title("F1 during active learning")
    plt.xlabel("Dataset len")

    plt.plot(
        epoch_dynamic, train_f1s,
        color="red",
        label="train set"
    )

    plt.plot(
        epoch_dynamic, val_f1s,
        color="blue",
        label="valid set"
    )

    plt.legend(loc="upper left")

    plt.tight_layout()

    if coef == None:
        plt.savefig("plots/active_f1_i" + str(init) + "_s" + str(step) + pref + ".png")
    else:
        plt.savefig("plots/active_f1_i" + str(init) + "_coef.png")

def plot_comparison_active(init=100, step=100, suff="active_"):
    pref = "_rand"

    train_losses_r = load_obj(suff + "train_loss_list" + pref)
    train_f1s_r = load_obj(suff + "train_f1_list" + pref)

    val_losses_r = load_obj(suff + "val_loss_list" + pref)
    val_f1s_r = load_obj(suff + "val_f1_list" + pref)

    train_losses = load_obj(suff + "train_loss_list")
    train_f1s = load_obj(suff + "train_f1_list")

    val_losses = load_obj(suff + "val_loss_list")
    val_f1s = load_obj(suff + "val_f1_list")

    total_epochs = len(train_losses) * step + init

    plt.figure(7)

    plt.ylabel("Loss")
    plt.title("Loss during active learning")
    plt.xlabel("Dataset len")

    plt.plot(
        range(init, total_epochs, step), train_losses,
        color="red",
        label="NSE on train"
    )

    plt.plot(
        range(init, total_epochs, step), val_losses,
        color="blue",
        label="NSE on valid"
    )

    plt.plot(
        range(init, total_epochs, step), train_losses_r,
        color="orange",
        label="random on train"
    )

    plt.plot(
        range(init, total_epochs, step), val_losses_r,
        color="dodgerblue",
        label="random on valid"
    )

    plt.legend(loc="upper left")

    plt.tight_layout()

    plt.savefig("plots/active_comp_loss_i" + str(init) + "_s" + str(step) + ".png")

    plt.figure(8)

    plt.ylabel("F1")
    plt.title("F1 during active learning")
    plt.xlabel("Dataset len")

    plt.plot(
        range(init, total_epochs, step), train_f1s,
        color="red",
        label="NSE on train"
    )

    plt.plot(
        range(init, total_epochs, step), val_f1s,
        color="blue",
        label="NSE on valid"
    )

    plt.plot(
        range(init, total_epochs, step), train_f1s_r,
        color="orange",
        label="random on train"
    )

    plt.plot(
        range(init, total_epochs, step), val_f1s_r,
        color="dodgerblue",
        label="random on valid"
    )

    plt.legend(loc="lower right")

    plt.tight_layout()

    plt.savefig("plots/active_comp_f1_i" + str(init) + "_s" + str(step) + ".png")

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

    plt.legend(loc="lower right")

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

def plot_hist(data):
    plt.figure(9)

    plt.ylabel("Count")
    plt.xlabel("Sentence length")

    bin_num = len(set(data))
    plt.hist(data, bins=bin_num, rwidth=0.9)

    plt.tight_layout()

    plt.savefig("plots/hist.png")

def plot_sent_entropies(x, y):
    plt.figure(10)

    plt.scatter(x, y, s=[1]*len(x), marker="o")

    plt.ylabel("Entropy")
    plt.xlabel("Sentence length")

    lens = defaultdict(int)
    cnts = defaultdict(int)
    for xi, yi in zip(x, y):
        lens[xi] += yi
        cnts[xi] += 1

    x = sorted(lens.keys())
    y = [lens[xi] / cnts[xi] for xi in x]

    plt.plot(x, y, color="red", label="average")
    plt.legend(loc="lower right")

    plt.tight_layout()

    plt.savefig("plots/entropy.png")

"""
def animate_entropy(data, suff="", pref=""):
    fig = plt.figure(1010)
    camera = Camera(fig)

    for i, (x, y) in enumerate(data):
        plt.scatter(x, y, s=[1]*len(x), marker="o")

        plt.title("Entropy distribution after training", i, "model")

        plt.ylabel("Entropy")
        plt.xlabel("Sentence length")

        lens = defaultdict(int)
        cnts = defaultdict(int)
        for xi, yi in zip(x, y):
            lens[xi] += yi
            cnts[xi] += 1

        x = sorted(lens.keys())
        y = [lens[xi] / cnts[xi] for xi in x]

        plt.plot(x, y, color="red", label="average")
        plt.legend(loc="lower right")

        plt.tight_layout()

        camera.snap()

    animation = camera.animate()  
    animation.save("plots/" + suff + "entropy" + pref + ".gif", writer="imagemagick")"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--since", type=int, default=0)

    args = parser.parse_args()

    plot_last_run()
    plot_in_comparison(args.since)