# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import argparse
from copy import deepcopy
from functools import partial
import json
import logging
import os
import sys
from pathlib import Path
from typing import List, Optional, Dict

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.nn.parallel import DistributedDataParallel
from fvcore.common.checkpoint import Checkpointer, PeriodicCheckpointer
from tqdm import tqdm
import torch.nn.functional as F

from dinov2.data import SamplerType, make_data_loader, make_dataset
from dinov2.data.transforms import make_classification_eval_transform, make_classification_train_transform
import dinov2.distributed as distributed
from dinov2.eval.metrics import MetricType, build_metric
from dinov2.eval.setup import get_args_parser as get_setup_args_parser
from dinov2.eval.setup import setup_and_build_model
from dinov2.eval.utils import ModelWithIntermediateLayers, evaluate
from dinov2.logging import MetricLogger
from dinov2.data.datasets.other_datasets import load_datasets

logger = logging.getLogger("dinov2")


def get_args_parser(
    description: Optional[str] = None,
    parents: Optional[List[argparse.ArgumentParser]] = None,
    add_help: bool = True,
):
    parents = parents or []
    setup_args_parser = get_setup_args_parser(parents=parents, add_help=False)
    parents = [setup_args_parser]
    parser = argparse.ArgumentParser(
        description=description,
        parents=parents,
        add_help=add_help,
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="cifar10"
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default="./data"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Batch Size (per GPU)",
        default=128,
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        help="Number de Workers",
        default=8
    )
    parser.add_argument('--metric', type=str, default='top1', choices=["top1", 'class-avg', "r2"])

    return parser

def build_step(X, Y, classifier, optimizer, w, criterion_fn):
    def step():
        optimizer.zero_grad()
        #assert False, [t.device for t in [
        #    X, Y,
        #    classifier.weight, classifier.bias
        #]]
        loss = criterion_fn(classifier(X), Y, reduction='sum')
        for p in classifier.parameters():
            loss = loss + p.pow(2).sum().mul(w)
        loss.backward()
        return loss
    return step

def l1_criterion_fn(normalize_input: bool=False, normalize_target: bool=False):
    def fn(input, target, **kwargs):
        if normalize_input:
            input = nn.functional.normalize(input, dim=1)
        if normalize_target:
            target = nn.functional.normalize(target, dim=1)

        return F.l1_loss(input, target, **kwargs)

    return fn

def r2_fn(normalize_input: bool=False, normalize_target: bool=False):

    def r2_score(y, x):
        """https://github.com/ruchikachavhan/amortized-invariance-learning-ssl/blob/f832c17ce3d59c7a16cfba3caeac4438034cab23/r2score.py#L4"""
        if normalize_input:
            x = nn.functional.normalize(x, dim=1)
        if normalize_target:
            y = nn.functional.normalize(y, dim=1)

        print("r2 pre reshape", x.shape, y.shape)
        x = x.flatten().detach().cpu().numpy()
        y = y.flatten().detach().cpu().numpy()


        A = np.vstack([x, np.ones(len(x))]).T

        print("r2: x, y, A", x.shape, y.shape, A.shape)
        # Use numpy's least squares function
        m, c = np.linalg.lstsq(A, y)[0]

        # print(m, c)
        # 1.97 -0.11

        # Define the values of our least squares fit
        f = m * x + c

        # print(f)
        # [ 1.86  3.83  5.8   7.77  9.74]

        # Calculate R^2 explicitly
        yminusf2 = (y - f)**2
        sserr = sum(yminusf2)
        mean = float(sum(y)) / float(len(y))
        yminusmean2 = (y - mean)**2
        sstot = sum(yminusmean2)
        R2 = 1. -(sserr / sstot)
        return R2

    return r2_score

def compute_accuracy(X, Y, classifier, metric_name_or_fn):
    with torch.no_grad():

        preds = classifier(X)
        if metric_name_or_fn in ["top1", "class-avg"]:
            preds = preds.argmax(1)

        if metric_name_or_fn == 'top1':
            acc = (preds == Y).float().mean().item()
        elif metric_name_or_fn == 'class-avg':
            total, count = 0., 0.
            for y in range(0, Y.max().item()+1):
                masks = Y == y
                if masks.sum() > 0:
                    total += (preds[masks] == y).float().mean().item()
                    count += 1
            acc = total / count

        else:
            assert not isinstance(metric_name_or_fn, str)
            acc = metric_name_or_fn(Y, preds)

    return acc
def test_linear(
    num_classes: int,
    embeddings: Dict[str, Dict[str, torch.Tensor]],
    metric: str
):
    if metric in ["top1", 'class-avg']:
        criterion_fn = F.cross_entropy
        metric = args.metric
    elif args.metric == "r2":
        criterion_fn = l1_criterion_fn()
        metric = r2_fn()
    
    classifier = nn.Linear(embeddings["train"]["features"].shape[1], num_classes).cuda()
    optim_kwargs = {
        'line_search_fn': 'strong_wolfe',
        'max_iter': 5000,
        'lr': 1.,
        'tolerance_grad': 1e-10,
        'tolerance_change': 0,
    }


    best_acc = 0.
    best_w = 0.
    best_classifier = None

    X_train = embeddings["train"]["features"].cuda()
    Y_train = embeddings["train"]["labels"].cuda()
    
    X_val = embeddings["val"]["features"].cuda()
    Y_val = embeddings["val"]["labels"].cuda()

    X_test = embeddings["test"]["features"].cuda()
    Y_test = embeddings["test"]["labels"].cuda()

    X_trainval = embeddings["trainval"]["features"].cuda()
    Y_trainval = embeddings["trainval"]["labels"].cuda()

    for w in tqdm(torch.logspace(-6, 5, steps=45).tolist(), "Gridsearch over w"):
        optimizer = optim.LBFGS(classifier.parameters(), **optim_kwargs)
        
        # assert False, [t.device for t in [
        #     X_train, Y_train, X_val, Y_val, X_test, Y_test, X_trainval, Y_trainval,
        #     classifier.weight, classifier.bias
        # ]]
        optimizer.step(
            build_step(X_train, Y_train, classifier, optimizer, w, criterion_fn=criterion_fn))
        acc = compute_accuracy(X_val, Y_val, classifier, metric)

        if best_acc < acc:
            best_acc = acc
            best_w = w
            best_classifier = deepcopy(classifier)

    print(f'BEST: w={best_w:.4e}, acc={best_acc:.4f}')

    optimizer = optim.LBFGS(best_classifier.parameters(), **optim_kwargs)
    optimizer.step(build_step(X_trainval, Y_trainval, best_classifier, optimizer, best_w, criterion_fn=criterion_fn))
    acc = compute_accuracy(X_test, Y_test, best_classifier, metric_name_or_fn=metric)
    return acc

def run_dump_embeddings(
    model,
    output_dir,
    dataset_name: str,
    dataset_root: str,
    batch_size,
    num_workers,
    autocast_dtype,
):
    seed = 0

    datasets = load_datasets(
        dataset=dataset_name,
        datadir=dataset_root,
        pretrain_data="imagenet100",
    )
    training_num_classes = datasets["num_classes"]
    sampler_type = None #SamplerType.SHARDED_INFINITE

    n_last_blocks_list = [1, 4]
    n_last_blocks = max(n_last_blocks_list)
    autocast_ctx = partial(torch.cuda.amp.autocast, enabled=True, dtype=autocast_dtype)
    feature_model = ModelWithIntermediateLayers(model, n_last_blocks, autocast_ctx)
    # sample_output = feature_model(datasets["train"][0][0].unsqueeze(0).cuda())

    result = dict()
    with torch.no_grad():
        for key in ["train", "val", "trainval", "test"]:

            all_embeddings = []
            all_labels = []

            loader = make_data_loader(
                dataset=datasets[key],
                batch_size=batch_size,
                num_workers=num_workers,
                shuffle=False,
                seed=seed,
                sampler_type=sampler_type,
                sampler_advance=0,
                drop_last=False,
                persistent_workers=True,
            )

            for data, labels in tqdm(loader, desc=f"{dataset_name}: {key}"):
                data = data.cuda(non_blocking=True)

                if data.ndim == 5:
                    _, n, c, h, w = data.shape
                    data = data.view(-1, c, h, w)
                    labels = labels.view(-1, 1).repeat(1, n).view(-1)

                features = model(data)

                features = features.detach().cpu()
                all_embeddings.append(features)
                all_labels.append(labels)
                #break 


            result[key] = {
                "features": torch.cat(all_embeddings, dim=0),
                "labels": torch.cat(all_labels, dim=0)
            }
            print(key)
            print({
                k: v.shape
                for k, v in result[key].items()
            })

    print(f"Saving embeddings")
    torch.save(result, Path(output_dir) / f"{dataset_name}_for_linear.pth")
    print("Test linear")
    linear_result = test_linear(num_classes=training_num_classes, embeddings=result, metric=args.metric)
    print(f"{dataset_name} test_linear/{args.metric}", linear_result)
    result["test_linear/{args.metric}"] = linear_result
    torch.save(result, Path(output_dir) / f"{dataset_name}_for_linear.pth")





def main(args):
    model, autocast_dtype = setup_and_build_model(args)
    run_dump_embeddings(
        model=model,
        dataset_name=args.dataset_name,
        dataset_root=args.dataset_root,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        autocast_dtype=autocast_dtype,
    )
    return 0


if __name__ == "__main__":
    description = "DINOv2 dump embeddings"
    args_parser = get_args_parser(description=description)
    args = args_parser.parse_args()
    sys.exit(main(args))
