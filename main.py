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
    nargs="?",
    const=None,  # no index
    type=int,  # optional index
    help="CIFAR dataset. Without index for whole dataset; with index for single image.",
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
compute_group.add_argument(
    "--bcomp",
    nargs="?",
    const=None,
    type=int,
    help="Batch comparison. No number = whole dataset; number = random subset size.",
)


def get_dataset_choice(args):
    if hasattr(args, "cifar") and args.cifar is not None:
        return "cifar", args.cifar

    if args.image is not None:
        return "image", args.image

    return None, None


parser.set_defaults(index=None)
args = parser.parse_args()

dataset, ds_value = get_dataset_choice(args)

if dataset == "cifar":
    args.index = ds_value  # int 或 None
elif dataset == "image":
    args.index = None  # custom image has no index

single_image_mode = dataset == "image" or isinstance(ds_value, int)
batch_mode = dataset != "image" and ds_value is None

#  Illegal combinations detection
if single_image_mode and args.bcomp is not None:
    parser.error(
        "ERROR: --bcomp cannot be used with a single image index "
        "(or with --image). Use --met or --comp instead."
    )

if batch_mode and (args.method or args.comp):
    parser.error(
        "ERROR: Batch mode requires --bcomp. "
        "Do not use --met or --comp when processing the entire dataset."
    )

if dataset == "image" and args.bcomp is not None:
    parser.error(
        "ERROR: --image cannot be used with --bcomp. "
        "Custom images support only --met or --comp."
    )


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


def print_iter(history, name):
    if history:
        cols = 10
        rows = max(1, math.ceil(len(history) / cols))
        plt.figure(figsize=(12, 4 * rows))
        for i, img in enumerate(history):
            plt.subplot(rows, cols, i + 1)
            plt.imshow(img)
            plt.title(f"iter={i * 10}")
            plt.axis("off")

        out_dir = Path("result")
        if args.bcomp is not None:
            out_dir = Path(f"result/{dataset}_bcomp")
        elif args.comp:
            out_dir = Path(f"result/{dataset}_comp")
        elif args.index is not None:
            out_dir = Path(f"result/{dataset}_index")

        out_dir.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(out_dir / f"{name}.png")
        plt.close()


def log_result(res, name):
    log_dir = Path("result")
    if args.bcomp is not None:
        log_dir = Path(f"result/{dataset}_bcomp")
    elif args.comp:
        log_dir = Path(f"result/{dataset}_comp")
    elif args.index is not None:
        log_dir = Path(f"result/{dataset}_index")

    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "result.log"

    mse_last = res["mses"][-1] if res.get("mses") else None
    input_ref = args.image if args.image else f"{dataset}_{args.index}"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(
            f"[{timestamp}] run={name} input={input_ref} final_loss={res['final_loss']}  "
            f"pred_label={res['pred_label']} mses_last={mse_last}\n"
        )


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
    Path(f"result/{dataset}_comp").mkdir(parents=True, exist_ok=True)
    if args.image:
        plt.savefig(f"result/Image_comp/{Path(args.image).stem}_comparison.png")
    else:
        plt.savefig(f"result/{dataset}_comp/{ args.index }_comparison.png")

    if args.image:
        print_iter(res_dlg["history"], f"{Path(args.image).stem}_DLG")
        print_iter(res_idlg["history"], f"{Path(args.image).stem}_iDLG.png")
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

elif args.bcomp is not None:

    print("\n================ Batch Comparison: DLG vs iDLG ================\n")

    total = len(dst)
    if args.bcomp is None:
        indices = list(range(total))
        print(f"Using full dataset: {total} samples")
    else:
        indices = list(np.random.choice(total, min(args.bcomp, total), replace=False))
        print(f"Randomly sampled {len(indices)} samples")

    mses_dlg, mses_idlg = [], []

    for idx in indices:
        args.index = idx
        gt_data = tp(dst[idx][0]).to(device).unsqueeze(0)
        gt_label = torch.tensor([dst[idx][1]]).to(device)
        gt_onehot_label = label_to_onehot(gt_label)

        # Build model + gradient
        net = LeNet().to(device)
        net.apply(weights_init)
        pred = net(gt_data)
        loss = cross_entropy_for_onehot(pred, gt_onehot_label)
        dy_dx = torch.autograd.grad(loss, net.parameters())
        original_dy_dx = [_.detach().clone() for _ in dy_dx]

        res_dlg = run_DLG()
        res_idlg = run_iDLG()

        mses_dlg.append(res_dlg["mses"][-1])
        mses_idlg.append(res_idlg["mses"][-1])

        log_result(res_dlg, "DLG")
        log_result(res_idlg, "iDLG")
        print_iter(res_dlg["history"], f"{dataset}_{idx}_DLG")
        print_iter(res_idlg["history"], f"{dataset}_{idx}_iDLG")

    # 统计 Fidelity 分布并绘图
    thresholds = [0.01, 0.005, 0.001, 0.0005, 0.0001]
    result_dlg = [
        100 * sum(m <= t for m in mses_dlg) / len(mses_dlg) for t in thresholds
    ]
    result_idlg = [
        100 * sum(m <= t for m in mses_idlg) / len(mses_idlg) for t in thresholds
    ]

    plt.figure()
    plt.plot(thresholds, result_dlg, marker="o", label="DLG")
    plt.plot(thresholds, result_idlg, marker="*", label="iDLG")
    plt.xlabel("Fidelity Threshold (MSE)")
    plt.ylabel("% of Good Fidelity")
    plt.title(f"{dataset.upper()}")
    plt.legend()
    plt.gca().invert_xaxis()
    Path(f"result/{dataset}_bcomp").mkdir(parents=True, exist_ok=True)
    plt.savefig(f"result/{dataset}_bcomp/fidelity.png")
    plt.close()
    exit()


plt.show()
