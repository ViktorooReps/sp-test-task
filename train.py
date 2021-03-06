from utils.memory_management import limit_memory, load_obj, save_obj
from utils.plotter import plot_mini, plot_hist, plot_sent_entropies
from utils.logging import Logger
from utils.reproducibility import seed_worker, seed
from model.nerc import *
from extract import preprocess

from pprint import pprint
from math import sqrt
from torch.utils.data import DataLoader, BatchSampler, SequentialSampler
from torch.nn.utils import clip_grad_norm_

from model.hyperparams import *
from evaluate import evaluate_model, evaluate_entropy

import argparse
import torch
import torch.optim as optim
import datetime

def train_epoch(model, dataloader, scheduler, optimizer):
    model.train()
    for batch_idx, (chars, toks, lbls, seq_lens) in enumerate(dataloader):
        chars = chars.to(device)
        toks = toks.to(device)
        lbls = lbls.to(device)

        emissions = model(chars, toks, seq_lens).to(device)
        loss = model.loss(emissions, lbls, seq_lens)

        loss.backward()

        clip_grad_norm_(model.parameters(), clipping_value)

        optimizer.step()
        optimizer.zero_grad()
    
    scheduler.step()

def fit_model(model, train_data, scheduler, optimizer, dl_args, epochs=10, val_data=None, 
    stopper=None, skip_train_eval=False, skip_val_eval=False, verbose=True):
    """Fits model to train_data. Uses early stopping if stopper isn't None"""

    train_losses = []
    train_f1s = []
    val_losses = []
    val_f1s = []

    if verbose:
        print("\nPre train results:")

    if not skip_train_eval:
        train_loss, train_f1 = evaluate_model(model, get_eval_dataloader(train_data, **dl_args))
        train_losses.append(train_loss)
        train_f1s.append(train_f1)

        if verbose:
            print("[train] loss: " + str(train_loss) + " F1: " + str(train_f1))
    
    if val_data != None and not skip_val_eval:
        val_loss, val_f1 = evaluate_model(model, get_eval_dataloader(val_data, **dl_args))
        val_losses.append(val_loss)
        val_f1s.append(val_f1)

        if stopper != None:
            stopper.add_epoch(model, val_f1)

        if verbose:
            print("[valid] loss: " + str(val_loss) + " F1: " + str(val_f1))

    print("\nTraining start")
    print(datetime.datetime.now())

    if stopper != None: # assume val_data isn't None
        epoch = 1
        while not stopper.stop(): 
            if verbose:
                print("\nTraining epoch " + str(epoch))

            train_epoch(
                model, get_train_dataloader(train_data, **dl_args),
                scheduler, optimizer
            )

            if not skip_train_eval:
                train_loss, train_f1 = evaluate_model(model, 
                    get_eval_dataloader(train_data, **dl_args))
                train_losses.append(train_loss)
                train_f1s.append(train_f1)

                if verbose:
                    print("[train] loss: " + str(train_loss) + " F1: " + str(train_f1))

            val_loss, val_f1 = evaluate_model(model, get_eval_dataloader(val_data, **dl_args))
            val_losses.append(val_loss)
            val_f1s.append(val_f1)

            if verbose:
                print("[valid] loss: " + str(val_loss) + " F1: " + str(val_f1))

            stopper.add_epoch(model, val_f1)
            epoch += 1

        best_model = stopper.get_model()
    else:
        for epoch in range(epochs): 
            if verbose:
                print("\nTraining epoch " + str(epoch))

            train_epoch(
                model, get_train_dataloader(train_data, **dl_args),
                scheduler, optimizer
            )

            train_loss, train_f1 = evaluate_model(model, get_eval_dataloader(train_data, **dl_args))
            train_losses.append(train_loss)
            train_f1s.append(train_f1)

            if verbose:
                print("[train] loss: " + str(train_loss) + " F1: " + str(train_f1))

            if val_data != None:
                val_loss, val_f1 = evaluate_model(model, get_eval_dataloader(val_data, **dl_args))
                val_losses.append(val_loss)
                val_f1s.append(val_f1)

                if verbose:
                    print("[valid] loss: " + str(val_loss) + " F1: " + str(val_f1))

        best_model = model

    return best_model, train_losses, train_f1s, val_losses, val_f1s

def fit_model_active(model_args, scheduler_args, optimizer_args, inflating_data, dl_args, val_data, 
    train_data, stopper, exclude_tags=set(), request_seqs=100, random_sampling=False, coef=None):

    inflating_data.train()

    logger = Logger()
    logger.register("labeled", "tagged", "seqs_num", "train_loss", "train_f1", "val_loss", "val_f1", "hist_data")
    logger.labeled = 0
    logger.tagged = 0
    logger.hist_data = (0, 0)
    logger.seqs_num = len(inflating_data)

    while not stopper.stop():
        inflating_data.train()
        print("\nFitting model number " + str(logger.times_written) 
            + " on dataset of len " + str(len(inflating_data))
            + " with stored left " + str(len(inflating_data.stored)))

        model = CNNbLSTMCRF(**model_args)

        optimizer = optim.SGD(model.parameters(), **optimizer_args)
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, **scheduler_args)
        model_stopper = EarlyStopper(prefix="cached_intermediate_", tolerance=5)
        best_model, _, _, _, _, = fit_model(model, inflating_data, scheduler, optimizer, dl_args, 
            val_data=val_data, stopper=model_stopper, skip_train_eval=True, verbose=False)

        inflating_data.eval()

        logger.train_loss, logger.train_f1 = evaluate_model(model, get_eval_dataloader(train_data, **dl_args))
        print("[train] loss: " + str(logger.train_loss) + " F1: " + str(logger.train_f1))

        logger.val_loss, logger.val_f1 = evaluate_model(model, get_eval_dataloader(val_data, **dl_args))
        print("[valid] loss: " + str(logger.val_loss) + " F1: " + str(logger.val_f1))

        inflating_data.entropy()

        if coef != None:
            request_seqs = int(len(inflating_data.seqs) * coef)

        if not random_sampling:
            entropies = evaluate_entropy(model, get_entropy_dataloader(inflating_data, **dl_args))

            # saving data for gif histogram
            x = []
            y = []
            for ind, entropy in enumerate(entropies):
                x.append(len(inflating_data.stored[ind][0]))
                y.append(entropy)
            logger.hist_data = (x, y)

            new_inds = np.array(entropies).argsort()[-request_seqs:][::-1]
        else:
            sampler = iter(RandomSampler(inflating_data))
            new_inds = set(next(sampler) for i in range(request_seqs))

        new_labeled = 0
        new_tagged = 0
        for ind in new_inds:
            new_labeled += len(inflating_data.stored[ind][0])
            for tag_idx in inflating_data.stored[ind][1]:
                if tag_idx not in exclude_tags:
                    new_tagged += 1

        inflating_data.add_seqs(new_inds)

        print("New labeled tokens:", new_labeled)
        print("New without O tag:", new_tagged)

        logger.write()

        logger.labeled += new_labeled
        logger.tagged += new_tagged
        logger.seqs_num += len(new_inds)

        stopper.add_epoch(best_model, logger.val_f1)

    print("\nLabeled token statistics")
    print("Total labeled:", logger.labeled)
    print("Average added sequence length:", logger.labeled / logger.seqs_num)

    best_model = stopper.get_model()
    return best_model, logger

if __name__ == '__main__':
    limit_memory(7 * 1024 * 1024 * 1024)
    seed()

    parser = argparse.ArgumentParser()
    parser.add_argument("--mini", action="store_true")
    parser.add_argument("--active", action="store_true")
    parser.add_argument("--randsampling", action="store_true")
    parser.add_argument("--stopper", action="store_true")

    args = parser.parse_args()

    char_to_idx = load_obj("char_to_idx")
    tag_to_idx = load_obj("tag_to_idx")
    tok_to_idx = load_obj("tok_to_idx")
    idx_to_tok = load_obj("idx_to_tok")

    token_voc = load_obj("token_voc")

    tok_to_emb = load_obj("tok_to_emb")
    oov_voc = load_obj("oov_voc")

    token_vecs = []
    for idx in range(len(tok_to_idx)):
        token = idx_to_tok[idx]
        if token not in oov_voc:
            token_vecs.append(tok_to_emb[token])
        else:
            r1 = - 1 / sqrt(token_emb_size)
            r2 = 1 / sqrt(token_emb_size)
            token_vecs.append((r1 - r2) * torch.rand(1, token_emb_size) + r2)

    token_vecs = torch.cat(token_vecs, dim=0)

    model_args = dict(
        char_to_idx=char_to_idx,
        tok_to_idx=tok_to_idx,
        tag_to_idx=tag_to_idx,
        token_vecs=token_vecs
    )
    
    lr_lambda = lambda x: 1 / (1 + decay_rate * x)
    optimizer_args = dict(
        lr=initial_lr,
        momentum=momentum
    )
    scheduler_args = dict(
        lr_lambda=lr_lambda
    )

    data_args = dict(
        char_to_idx=char_to_idx, 
        tok_to_idx=tok_to_idx,
        tag_to_idx=tag_to_idx,
        max_token_len=max_word_len,
        preprocessor=preprocess
    )

    mini_seqs = load_obj("mini_seqs")
    mini_data = Data(mini_seqs, active=args.active, **data_args)

    train_seqs = load_obj("train_seqs")
    train_data = Data(train_seqs, active=args.active, **data_args)

    val_seqs = load_obj("val_seqs")
    val_data = Data(val_seqs, **data_args)

    train_loss_list = []
    val_loss_list = []

    train_f1_list = []
    val_f1_list = []

    mini_loss_list = []
    mini_f1_list = []

    dl_args = dict(
        batch_size=batch_size,
        worker_init_fn=seed_worker,
        pad_collator=PaddingCollator(
            char_pad=char_to_idx["<pad>"],
            max_word_len=max_word_len,
            tok_pad=tok_to_idx["<pad>"],
            tag_pad=tag_to_idx["O"]
        )
    )

    if args.active:
        inflating_data = Data(train_seqs, active=True, starting_size=starting_size, **data_args)
        stopper = EarlyStopper(min_epochs=global_min_epochs, max_epochs=global_max_epochs,
            tolerance=global_tolerance
        )

        suff = "active_"
        if args.randsampling:
            pref = "_rand"
        else:
            pref = ""

        try:
            best_model, logger = fit_model_active(
                model_args, scheduler_args, optimizer_args, inflating_data, dl_args, val_data, 
                train_data, stopper, request_seqs=request_seqs, random_sampling=args.randsampling,
                coef=coef, exclude_tags={"O"}
            )
            save_obj(logger, suff + "logger" + pref)
        except KeyboardInterrupt:
            print("\nTraining interrupted! Restoring best model achieved so far...")
            best_model = stopper.get_model()

        save_obj(best_model, suff + "model" + pref)
        
    else:
        model = CNNbLSTMCRF(char_to_idx, tok_to_idx, tag_to_idx, token_vecs)

        print("Trainable weights:")
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(name, param.data.shape, "norm:", torch.norm(param.data).item())

        optimizer = optim.SGD(model.parameters(), **optimizer_args)
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, **scheduler_args)

        if args.mini:
            _, mini_loss_list, mini_f1_list, _, _ = fit_model(
                model, mini_data, scheduler, optimizer, dl_args, epochs=epochs
            )

            save_obj(mini_loss_list, "mini_loss_list")
            save_obj(mini_f1_list, "mini_f1_list")

            plot_mini()
        else:
            if args.stopper:
                stopper = EarlyStopper(min_epochs=global_min_epochs, max_epochs=global_max_epochs,
                    tolerance=global_tolerance
                )
                best_model, train_loss_list, train_f1_list, val_loss_list, val_f1_list = fit_model(
                    model, train_data, scheduler, optimizer, dl_args, val_data=val_data, stopper=stopper 
                )
            else:
                best_model, train_loss_list, train_f1_list, val_loss_list, val_f1_list = fit_model(
                    model, train_data, scheduler, optimizer, dl_args, epochs=epochs, val_data=val_data
                )

            save_obj(train_loss_list, "train_loss_list")
            save_obj(val_loss_list, "val_loss_list")

            save_obj(train_f1_list, "train_f1_list")
            save_obj(val_f1_list, "val_f1_list")

            save_obj(best_model, "model")
