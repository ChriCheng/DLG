# -*- coding: utf-8 -*-
import argparse
import math
import os
import numpy as np
from pprint import pprint
from pathlib import Path
from datetime import datetime
from PIL import Image
import matplotlib.pyplot as plt

import torch

import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad
import torchvision
from torchvision import models, datasets, transforms


print(torch.__version__, torchvision.__version__)

from utils import label_to_onehot, cross_entropy_for_onehot

parser = argparse.ArgumentParser(description="Deep Leakage from Gradients.")

datasets_group = parser.add_mutually_exclusive_group(required=True)

datasets_group.add_argument(
    "--cifar",
    type=int,
    dest="index",
    help="the index for leaking images on CIFAR.",
)
datasets_group.add_argument("--image", type=str, help="the path to customized image.")

compute_group = parser.add_mutually_exclusive_group(required=True)
compute_group.add_argument(
    "--met",
    type=str,
    choices=["DLG", "iDLG"],
    dest="method",
    help="Attack method: DLG or iDLG.",
)
compute_group.add_argument(
    "--comp",
    action="store_true",
    help="Compare DLG and iDLG on the same image.",
)
args = parser.parse_args()

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
else:
    torch.backends.mkldnn.enabled = False  # disable MKLDNN to get second-order grads on M2 CPU(in fact is Mac vision of PyTorch)

print("Running on %s" % device)
print(f"Using method: {args.method}")

dst = datasets.CIFAR100("~/.torch", download=True)  # your dataset path here
tp = transforms.ToTensor()  # transform to tensor
tt = transforms.ToPILImage()  # transform to PIL image

img_index = args.index


if args.image:
    # 1. read image
    img = Image.open(args.image).convert("RGB")
    target_size = (32, 32)

    folder, filename = os.path.split(args.image)
    name, ext = os.path.splitext(filename)
    resized_path = os.path.join(folder, f"{name}_resize{ext}")

    # 2. ensure size
    if img.size == target_size:
        img_for_model = img
        print(f"✅ The input size has been met :{target_size}.")
    elif os.path.exists(resized_path):
        img_for_model = Image.open(resized_path).convert("RGB")
        print(f"✅  Existing resized images :{resized_path}.")
    else:
        img_for_model = img.resize(target_size)
        img_for_model.save(resized_path)
        print(f"✅ Resized pic saving at :{resized_path}")

    # 3. to tensor
    gt_data = tp(img_for_model).to(device)

    # 4. fake label
    gt_label = torch.Tensor([dst[25][1]]).long().to(device)
    print(f"⚠️  Using fake label: {dst[25][1]} for customized image.")


else:
    gt_data = tp(dst[img_index][0]).to(device)
    gt_label = torch.Tensor([dst[img_index][1]]).long().to(device)

gt_data = gt_data.view(1, *gt_data.size())


gt_label = gt_label.view(  # add batch size
    1,
)
gt_onehot_label = label_to_onehot(gt_label)

plt.figure("Ground Truth")
plt.imshow(tt(gt_data[0].cpu()))
plt.axis("off")

from models.vision import LeNet, weights_init

net = LeNet().to(device)


# torch.manual_seed(1234)

net.apply(weights_init)
criterion = cross_entropy_for_onehot
# criterion = nn.CrossEntropyLoss()  # for iDLG


# compute original gradient
pred = net(gt_data)
y = criterion(pred, gt_onehot_label)
dy_dx = torch.autograd.grad(y, net.parameters())

original_dy_dx = list((_.detach().clone() for _ in dy_dx))
# So this is FedSGD pattern where only one batch is used to compute gradient
# even just one epoch of training hhh

num_classes = gt_onehot_label.size(1)


def run_DLG():

    dummy_data = torch.randn(gt_data.size()).to(device).requires_grad_(True)
    dummy_label = torch.randn(gt_onehot_label.size()).to(device).requires_grad_(True)
    # data and label are following normal distribution(N(0,1)) to initialize

    optimizer = torch.optim.LBFGS(
        [dummy_data, dummy_label], line_search_fn="strong_wolfe"
    )
    # qutoted from DLG ‘We use L-BFGS [25] with learning rate 1,
    # history size 100 and max iterations 20 and optimize for 1200 iterations

    history = []
    mses = []  # record mse of dummy_data and gt_data
    recent_losses = (
        []
    )  # record loss(differencen between dummy gradient and original gradient)
    stop_iter = None
    final_loss = None

    last_grad_diff = None
    for iters in range(300):

        def closure():
            nonlocal last_grad_diff
            optimizer.zero_grad()

            dummy_pred = net(dummy_data)

            dummy_onehot_label = F.softmax(dummy_label, dim=-1)
            # apply softmax to make dummy_label one-hot like
            dummy_loss = criterion(dummy_pred, dummy_onehot_label)

            dummy_dy_dx = torch.autograd.grad(
                dummy_loss, net.parameters(), create_graph=True
            )

            grad_diff = 0  # gradient difference
            for gx, gy in zip(dummy_dy_dx, original_dy_dx):
                grad_diff += ((gx - gy) ** 2).sum()
            last_grad_diff = grad_diff.detach()
            grad_diff.backward()

            return grad_diff

        optimizer.step(closure)

        with torch.no_grad():  # record history and avoid unnecessary computation graph
            if iters % 10 == 0:
                dummy_pred = net(dummy_data)

                current_loss = last_grad_diff.item()  # grad_diff
                final_loss = current_loss
                current_mse = torch.mean((dummy_data - gt_data) ** 2).item()
                mses.append(current_mse)
                print(f"[DLG] iter {iters} loss = {current_loss} mses = {current_mse}")
                history.append(tt(dummy_data[0].detach().cpu()))
                recent_losses.append(current_loss)

                if current_loss < 0.000001:
                    break

    # predict label from dummy_label
    with torch.no_grad():
        dummy_onehot = F.softmax(dummy_label, dim=-1)
        pred_label = torch.argmax(dummy_onehot, dim=-1).item()

    return {
        "method": "DLG",
        "history": history,
        "final_loss": final_loss,
        "stop_iter": stop_iter,
        "pred_label": pred_label,
        "mses": mses,
    }


def run_iDLG():

    # 1. From the last layer to predict the labels (the core step)
    grad_last_weight = original_dy_dx[-2]  # The final layer of FC weight gradients

    label_pred = (
        torch.argmin(torch.sum(grad_last_weight, dim=-1), dim=-1)
        .detach()
        .reshape((1,))
        .requires_grad_(False)
    )
    # 2.generate dummy data
    dummy_data = torch.randn(gt_data.size()).to(device).requires_grad_(True)
    dummy_label = None  # iDLG don't need dummy_label

    # 3. initialize CrossEntropyLoss
    CE_loss = nn.CrossEntropyLoss()

    # 4. initialize optimizer(without strong_wolfe line search)
    optimizer = torch.optim.LBFGS([dummy_data])

    history = []
    recent_losses = []
    mses = []
    final_loss = None
    stop_iter = None

    last_grad_diff = None
    for iters in range(300):

        def closure():
            nonlocal last_grad_diff
            optimizer.zero_grad()

            dummy_pred = net(dummy_data)

            dummy_loss = CE_loss(dummy_pred, label_pred)

            dummy_dy_dx = torch.autograd.grad(
                dummy_loss, net.parameters(), create_graph=True
            )

            grad_diff = 0
            for gx, gy in zip(dummy_dy_dx, original_dy_dx):
                grad_diff += ((gx - gy) ** 2).sum()

            last_grad_diff = grad_diff.detach()
            grad_diff.backward()
            return grad_diff

        optimizer.step(closure)

        if iters % 10 == 0:
            with torch.no_grad():
                dummy_pred = net(dummy_data)
                # current_loss = CE_loss(dummy_pred, label_pred).item()
                # current_value = current_loss
                current_loss = last_grad_diff.item()  # grad_diff
                final_loss = current_loss
                current_mse = torch.mean((dummy_data - gt_data) ** 2).item()
                mses.append(current_loss)
                print(f"[DLG] iter {iters} loss = {current_loss} mses = {current_mse}")
                history.append(tt(dummy_data[0].detach().cpu()))
                recent_losses.append(current_loss)
                if current_loss < 0.000001:
                    break

    return {
        "method": "iDLG",
        "history": history,
        "final_loss": final_loss,
        "stop_iter": stop_iter,
        "pred_label": label_pred.item(),
        "mses": mses,
    }


def log_result(res, name):
    """Append key metrics to result/result.log."""
    log_dir = Path("result")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "result.log"
    mse_last = res["mses"][-1] if res.get("mses") else None
    input_ref = args.image if args.image else f"CIFAR index {args.index}"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with log_path.open("a", encoding="utf-8") as f:
        f.write(
            f"[{timestamp}] run={name} input={input_ref} "
            f"final_loss={res['final_loss']}  "
            f"pred_label={res['pred_label']} final_mses={mse_last}\n"
        )


def print_iter(history, name):
    if history:
        cols = 10
        rows = max(1, math.ceil(len(history) / cols))
        plt.figure(f"{name} Iteration", figsize=(12, 4 * rows))
        for i, snapshot in enumerate(history):
            plt.subplot(rows, cols, i + 1)
            plt.imshow(snapshot)
            plt.title("iter=%d" % (i * 10))
            plt.axis("off")
            if args.image:
                plt.savefig(rf"result/{Path(args.image).stem}_DLG.png")
            else:
                plt.savefig(f"result/{name}.png")

    else:
        print("No snapshots recorded; nothing to plot.")


if args.comp:
    print("\n================ Compare DLG vs iDLG ================\n")

    res_dlg = run_DLG()
    res_idlg = run_iDLG()

    print("\n------ DLG Result ------")
    print(f"final_loss = {res_dlg['final_loss']}")
    print(f"mses = {res_dlg['mses'][-1]}")
    print(f"pred_label = {res_dlg['pred_label']}")

    log_result(res_dlg, "DLG")

    print("\n------ iDLG Result ------")
    print(f"final_loss = {res_idlg['final_loss']}")
    print(f"mses = {res_idlg['mses'][-1]}")
    print(f"pred_label = {res_idlg['pred_label']}")

    log_result(res_idlg, "iDLG")

    plt.figure("Comparison", figsize=(10, 5))

    if res_dlg["history"]:
        plt.subplot(1, 2, 1)
        plt.imshow(res_dlg["history"][-1])
        plt.title(f"DLG\nloss={res_dlg['final_loss']:}")
        plt.axis("off")

    if res_idlg["history"]:
        plt.subplot(1, 2, 2)
        plt.imshow(res_idlg["history"][-1])
        plt.title(f"iDLG\nloss={res_idlg['final_loss']:}")
        plt.axis("off")
    if args.image:
        plt.savefig(f"result/{Path(args.image).stem}_comparison.png")
    else:
        plt.savefig(f"result/cifar_{ args.index }_comparison.png")

    if args.image:
        print_iter(res_dlg["history"], args.image + "_DLG")
        print_iter(res_idlg["history"], args.image + "_iDLG")
    else:
        print_iter(res_dlg["history"], f"cifar_{ args.index }_DLG")
        print_iter(res_idlg["history"], f"cifar_{ args.index }_iDLG")


elif args.method:
    if args.method == "DLG":
        res = run_DLG()
    elif args.method == "iDLG":
        res = run_iDLG()

    log_result(res, args.method)
    history = res["history"]

    if history:
        print_iter(history, args.method)

    else:
        print("No snapshots recorded; nothing to plot.")

    # show the last leaked image
    plt.figure(" Leaked images")
    plt.imshow(history[-1])
    plt.axis("off")

plt.show()
