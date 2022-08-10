# flake8: noqa
import argparse
from random import random
import time
import numpy as np
from flair.models import TARSClassifier
from flair.data import Corpus, Sentence
from flair.datasets import SentenceDataset
from flair.trainers import ModelTrainer
import torch
import flair
from pathlib import Path

torch.cuda.empty_cache()


def set_seed(seed):
    flair.set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


set_seed(42)

import json

with open(
    "grouped_data.json",
    encoding="utf-8",
) as fp:
    g_data = json.load(fp)

group_count = 1


def train_module(model, fine_tune, ff_dim, nhead, epoch, train_ds, test_ds):
    global group_count
    print(f"Model : {model}\n FF_DIM : {ff_dim}\nnHead : {nhead}")

    base_path = (
        f"taggers/clinc_{model}_ft_{fine_tune}_{ff_dim}_{nhead}_group_{group_count}"
    )
    if type(base_path) is str:
        base_path = Path(base_path)
    # intialize the train/ test corpus with the data provided
    corpus = Corpus(train=train_ds, test=test_ds)
    # initalize the model adding the provided head config
    bert = TARSClassifier(
        nhead=nhead,
        ff_dim=ff_dim,
        fine_tune=fine_tune,
    )
    # switches a task with new labels provided during dataset creation
    bert.add_and_switch_to_new_task(
        f"clinc_head_task_{group_count}",
        label_dictionary=corpus.make_label_dictionary(
            label_type=f"clinc_head_group_{group_count}"
        ),
        label_type=f"clinc_head_group_{group_count}",
    )
    # intializes the trainer
    trainer = ModelTrainer(bert, corpus)
    start_time = time.time()
    # starts training
    data = trainer.train(
        base_path=base_path,  # path to store the model artifacts
        learning_rate=0.02,
        mini_batch_size=16,
        max_epochs=epoch,
        monitor_train=False,  # if we want to monitor train change to True
        embeddings_storage_mode="cuda",
        train_with_dev=True,  # if false does evaluation after each epoch on dev dataset
        checkpoint=True,  # creates a checkpoint, can be used to train further
    )

    print(
        f"\n\nTime taken to complete the model training : {time.time()-start_time}\n\n"
    )
    print(data)
    group_count += 1


if __name__ == "__main__":
    # Initialize parser
    parser = argparse.ArgumentParser()

    # Adding optional argument
    parser.add_argument("-m", "--model", help="TARS/BERT", default="BERT")
    parser.add_argument(
        "-ft",
        "--fine_tune",
        help="Train the model (True/False)",
        type=bool,
        default=False,
    )
    parser.add_argument(
        "-dim",
        "--ffdim",
        help="Feedforward Dimension Size (2048/1024/512/256)",
        type=int,
        default=1024,
    )
    parser.add_argument(
        "-nh",
        "--nhead",
        help="Feedforward attention head numbers (8/4/2)",
        default=8,
        type=int,
    )
    parser.add_argument("-e", "--epoch", help="numbers of Epoch", default=5, type=int)

    # Read arguments from command line
    args = parser.parse_args()
    for group_key in g_data:
        train_ds = []
        test_ds = []
        # creates a train dataset format for flair
        for key in g_data[group_key]["train"].keys():
            for data in g_data[group_key]["train"][key]:
                train_ds.append(
                    Sentence(data.lower()).add_label(
                        f"clinc_head_group_{group_count}", key.lower()
                    )
                )
        # creates a test dataset format for flair
        for key in g_data[group_key]["test"].keys():
            for data in g_data[group_key]["test"][key]:
                test_ds.append(
                    Sentence(data.lower()).add_label(
                        f"clinc_head_group_{group_count}", key.lower()
                    )
                )
        # creates a train/test datasets
        train_ds = SentenceDataset(train_ds)
        test_ds = SentenceDataset(test_ds)
        print(train_ds[0])
        print(test_ds[0])
        print("data_load Completed")
        # invoking the training, with provided configuration and data
        train_module(
            model=args.model,
            fine_tune=args.fine_tune,
            ff_dim=args.ffdim,
            nhead=args.nhead,
            epoch=args.epoch,
            train_ds=train_ds,
            test_ds=test_ds,
        )
        if group_count == 4:
            break
