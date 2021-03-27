from model.hyperparams import *
from model.nerc import Data, get_dataloader
from utils.memory_management import load_obj, save_obj
from utils.plotter import plot_last_run, plot_in_comparison
from extract import preprocess

from torch.utils.data import DataLoader
from pprint import pprint 

import matplotlib.pyplot as plt
import numpy as np

def split_tag(tag):
    """Tag format: [B/I]-[PER/ORG/LOC/MISC] ID-type"""

    if tag == "O":
        return tag, None

    ent_id, ent_type = tag.split("-")

    return ent_id, ent_type

def assemble_entity(ent_type, idx_list):
    return tuple([ent_type] + idx_list)

def extract_entities(tags, idx_to_tag):
    """Extracts entities encoded in series of BIO tags
    
    Entity structure: ("type", B-tag, I-tag, I-tag, ... )
    B-tags and I-tags represent indicies of corressponding tags in tags list

    Returns set of extracted entities
    """

    extracted_entities = set()

    entity_assembler = { 
        "PER" : [],
        "ORG" : [],
        "LOC" : [],
        "MISC": [] 
    }

    for tag_idx, tag_id in enumerate(tags):
        tag = idx_to_tag[tag_id]
        ent_id, ent_type = split_tag(tag)

        if ent_type != None: 
            idx_list = entity_assembler[ent_type]

            if ent_id == "B":
                if len(idx_list) > 0:
                    extracted_entities.add(assemble_entity(ent_type, idx_list))
                
                entity_assembler[ent_type] = [tag_idx]

            if ent_id == "I":
                entity_assembler[ent_type] += [tag_idx]
    
    for ent_type in entity_assembler:
        idx_list = entity_assembler[ent_type]

        if len(idx_list) > 0:
            extracted_entities.add(assemble_entity(ent_type, idx_list))
    
    return extracted_entities

def score_entities(preds, labels, idx_to_tag):
    """Returns scores with respect to recognized named entities"""

    pred_entities = extract_entities(preds, idx_to_tag)
    true_entities = extract_entities(labels, idx_to_tag)

    TP = len(pred_entities.intersection(true_entities))
    FP = len(pred_entities.difference(true_entities))
    FN = len(true_entities.difference(pred_entities))

    with np.errstate(divide='ignore', invalid='ignore'):
        precision = np.true_divide(TP, (TP + FP))
        recall = np.true_divide(TP, (TP + FN))
        f1 = np.true_divide(2 * precision * recall, (precision + recall))

    return f1, precision, recall

def score_tokens(labels, preds, idx_to_tag, excluded_tags={"O"}):
    """Returns averaged F1 measure over tag classes and scores for each class"""

    tag_to_oh_preds = dict()
    tag_to_oh_lbls = dict()

    for val in idx_to_tag.values():
        tag_to_oh_preds[val] = []
        tag_to_oh_lbls[val] = []

    for pred in preds:
        for val in idx_to_tag.values():
            tag_to_oh_preds[val].append(1 if val == idx_to_tag[pred] else 0)

    for lbl in labels:
        for val in idx_to_tag.values():
            tag_to_oh_lbls[val].append(1 if val == idx_to_tag[lbl] else 0)

    tag_to_oh_preds = { val : torch.tensor(lst, requires_grad=False) 
        for val, lst in tag_to_oh_preds.items() }

    tag_to_oh_lbls = { val : torch.tensor(lst, requires_grad=False) 
        for val, lst in tag_to_oh_lbls.items() }

    TP = lambda lbl, prds: torch.sum(lbl * prds).item()
    FP = lambda lbl, prds: torch.sum((lbl == 0) * prds).item()
    FN = lambda lbl, prds: torch.sum(lbl * (prds == 0)).item()

    tag_to_cnts = {
        val : {
            "TP" : TP(tag_to_oh_lbls[val], tag_to_oh_preds[val]),
            "FP" : FP(tag_to_oh_lbls[val], tag_to_oh_preds[val]),
            "FN" : FN(tag_to_oh_lbls[val], tag_to_oh_preds[val])
        } for val in idx_to_tag.values()
    }

    with np.errstate(divide='ignore', invalid='ignore'):
        tag_to_score = {
            val : {
                "precision" : np.true_divide(tag_to_cnts[val]["TP"],
                    (tag_to_cnts[val]["TP"] + tag_to_cnts[val]["FP"])),
                "recall"    : np.true_divide(tag_to_cnts[val]["TP"],
                    (tag_to_cnts[val]["TP"] + tag_to_cnts[val]["FN"])),
                "f1"        : np.true_divide(tag_to_cnts[val]["TP"], 
                    (tag_to_cnts[val]["TP"] + (tag_to_cnts[val]["FP"] + tag_to_cnts[val]["FN"]) / 2))
            } for val in idx_to_tag.values()
        }

    f1 = sum(tag_to_score[val]["f1"] if val not in excluded_tags else 0 
        for val in idx_to_tag.values()) / (len(idx_to_tag) - len(excluded_tags))

    return f1, tag_to_score

def evaluate_model(model, dataloader):
    model.eval()

    predicted_labels = []
    labels = []
    total_batches = 0
    total_loss = 0
    
    with torch.no_grad():
        for batch_idx, (chars, toks, lbls, seq_lens) in enumerate(dataloader):
            chars = chars.to(device)
            toks = toks.to(device)
            lbls = lbls.to(device)

            emissions = model(chars, toks, seq_lens).to(device)
            loss = model.loss(emissions, lbls, seq_lens).item()

            total_loss += loss
            total_batches += 1

            predicted_labels += sum(model.decode(emissions, seq_lens), [])

            labels += sum(lbls.detach().tolist(), [])

        final_loss = total_loss / total_batches

        idx_to_tag = load_obj("idx_to_tag")
        f1, _, _ = score_entities(labels, predicted_labels, idx_to_tag)

    return (final_loss, f1)

if __name__ == '__main__':
    char_to_idx = load_obj("char_to_idx")
    tag_to_idx = load_obj("tag_to_idx")
    tok_to_idx = load_obj("tok_to_idx")

    model = load_obj("model")

    test_tokens = load_obj("test_tokens")
    test_labels = load_obj("test_labels")

    test_data = Data(test_tokens, test_labels, 
        tok_to_idx, char_to_idx, tag_to_idx, max_token_len=max_word_len,
        padding=padding, preprocessor=preprocess)

    train_tokens = load_obj("train_tokens")
    train_labels = load_obj("train_labels")

    train_data = Data(train_tokens, train_labels, 
        tok_to_idx, char_to_idx, tag_to_idx, max_token_len=max_word_len,
        padding=padding, preprocessor=preprocess)

    val_tokens = load_obj("val_tokens")
    val_labels = load_obj("val_labels")

    val_data = Data(val_tokens, val_labels, 
        tok_to_idx, char_to_idx, tag_to_idx, max_token_len=max_word_len,
        padding=padding, preprocessor=preprocess)

    print("\nEvaluating on test set")
    test_loss, test_f1 = evaluate_model(
        model, 
        get_eval_dataloader(test_data, batch_size, seq_len)
    )
    print("Evaluating on train set")
    train_loss, train_f1 = evaluate_model(
        model, 
        get_eval_dataloader(train_data, batch_size, seq_len)
    )
    print("Evaluating on valid set")
    val_loss, val_f1 = evaluate_model(
        model, 
        get_eval_dataloader(val_data, batch_size, seq_len)
    )
    print("\n[test]  loss: " + str(test_loss) + " F1: " + str(test_f1))
    print("[train] loss: " + str(train_loss) + " F1: " + str(train_f1))
    print("[valid] loss: " + str(val_loss) + " F1: " + str(val_f1))

    val_f1s = np.array(load_obj("val_f1_list"))
    print("Epochs with best F1 scores on validation set:")
    print(val_f1s.argsort()[::-1])

    plot_last_run()
    plot_in_comparison(5)