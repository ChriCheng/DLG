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


gt_label = gt_label.view(
    1,
)
gt_onehot_label = label_to_onehot(gt_label)

plt.figure("Ground Truth")
plt.imshow(tt(gt_data[0].cpu()))
plt.axis("off")

from models.vision import LeNet, weights_init

net = LeNet().to(device)


torch.manual_seed(1234)

net.apply(weights_init)
criterion = cross_entropy_for_onehot


# compute original gradient
pred = net(gt_data)
y = criterion(pred, gt_onehot_label)
dy_dx = torch.autograd.grad(y, net.parameters())

original_dy_dx = list(
    (_.detach().clone() for _ in dy_dx)
)  # detach the gradients to avoid unnecessary computation graph

# So this is FedSGD pattern where only one batch is used to compute gradient
# even just one epoch of training hhh

num_classes = gt_onehot_label.size(1)


def run_DLG():
    # generate dummy data and label
    dummy_data = torch.randn(gt_data.size()).to(device).requires_grad_(True)
    dummy_label = torch.randn(gt_onehot_label.size()).to(device).requires_grad_(True)
    # data and label are following normal distribution(N(0,1)) to initialize

    # nobody care about this dummy init image (
    # plt.figure("Dummy Init")
    # plt.imshow(tt(dummy_data[0].detach().cpu()))
    # plt.axis("off")

    optimizer = torch.optim.LBFGS(
        [dummy_data, dummy_label], line_search_fn="strong_wolfe"
    )
    # qutoted from DLG ‘We use L-BFGS [25] with learning rate 1,
    # history size 100 and max iterations 20 and optimize for 1200 iterations
    # and 100 iterations for image and text task respectively

    history = []
    mses = []
    # Early stop searching: stop when loss repeats exactly 3 times
    recent_losses = []
    plateau_patience = 3
    stop_iter = None
    final_loss = None

    for iters in range(300):

        def closure():
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
            grad_diff.backward()

            return grad_diff

        optimizer.step(closure)

        with torch.no_grad():  # record history and avoid unnecessary computation graph
            if iters % 10 == 0:
                dummy_pred = net(dummy_data)
                dummy_onehot = F.softmax(dummy_label, dim=-1)
                current_loss = criterion(dummy_pred, dummy_onehot)
                current_value = current_loss.item()
                final_loss = current_value
                current_mse = torch.mean((dummy_data - gt_data) ** 2).item()
                mses.append(current_loss)
                print(
                    f"[DLG] iter {iters} loss = {current_value:.4f} mses = {current_mse}"
                )
                history.append(tt(dummy_data[0].detach().cpu()))
                recent_losses.append(current_value)

                if len(recent_losses) >= plateau_patience:
                    window = recent_losses[-plateau_patience:]
                    if len(set(window)) == 1:
                        stop_iter = iters
                        print(
                            "[DLG] Loss stayed at %.4f for %d snapshots; stop optimizing."
                            % (current_value, plateau_patience)
                        )
                        break

    # 最终标签预测（softmax(dummy_label)）
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

    # 1. 从最后一层梯度预测标签（核心步骤）
    grad_last_weight = original_dy_dx[-2]  # 最后一层 FC 权重梯度
    # label_pred = torch.argmin(torch.sum(grad_last_weight, dim=-1)).detach()
    # label_pred = label_pred.view(1).long().to(device)
    label_pred = (
        torch.argmin(torch.sum(grad_last_weight, dim=-1), dim=-1)
        .detach()
        .reshape((1,))
        .requires_grad_(False)
    )
    # 2. 生成 dummy 数据
    dummy_data = torch.randn(gt_data.size()).to(device).requires_grad_(True)
    dummy_label = None  # iDLG 不优化 dummy_label

    # 3. iDLG 必须使用整数版 CrossEntropyLoss
    CE_loss = nn.CrossEntropyLoss()

    # 4. 标准 LBFGS（不要 strong_wolfe）
    optimizer = torch.optim.LBFGS([dummy_data])

    history = []
    recent_losses = []
    mses = []
    plateau_patience = 3
    final_loss = None
    stop_iter = None

    # 5. 迭代优化
    for iters in range(300):

        def closure():
            optimizer.zero_grad()

            # 前向
            dummy_pred = net(dummy_data)

            # CrossEntropyLoss(pred, integer_label)
            dummy_loss = CE_loss(dummy_pred, label_pred)

            # 计算新的 dummy 梯度
            dummy_dy_dx = torch.autograd.grad(
                dummy_loss, net.parameters(), create_graph=True
            )

            # 梯度距离（攻击核心）
            grad_diff = 0
            for gx, gy in zip(dummy_dy_dx, original_dy_dx):
                grad_diff += ((gx - gy) ** 2).sum()

            grad_diff.backward()
            return grad_diff

        optimizer.step(closure)

        # 记录可视化
        if iters % 10 == 0:
            with torch.no_grad():
                dummy_pred = net(dummy_data)
                current_loss = CE_loss(dummy_pred, label_pred).item()
                current_value = current_loss
                final_loss = current_loss
                current_mse = torch.mean((dummy_data - gt_data) ** 2).item()
                mses.append(current_loss)
                print(
                    f"[DLG] iter {iters} loss = {current_value:.4f} mses = {current_mse}"
                )
                history.append(tt(dummy_data[0].detach().cpu()))
                recent_losses.append(current_value)
                if len(recent_losses) >= plateau_patience:
                    window = recent_losses[-plateau_patience:]
                    if len(set(window)) == 1:
                        stop_iter = iters
                        print(
                            "[iDLG] Loss stayed at %.4f for %d snapshots; stop optimizing."
                            % (current_value, plateau_patience)
                        )
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
            f"final_loss={res['final_loss']} stop_iter={res['stop_iter']} "
            f"pred_label={res['pred_label']} mses_last={mse_last}\n"
        )


def IterPrint(history, name):
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
                plt.savefig(f"{Path(args.image).stem}_DLG.png")
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
    print(f"stop_iter  = {res_dlg['stop_iter']}")
    print(f"pred_label = {res_dlg['pred_label']}")
    print(f"mses = {res_dlg['mses'][-1]}")
    log_result(res_dlg, "DLG")

    print("\n------ iDLG Result ------")
    print(f"final_loss = {res_idlg['final_loss']}")
    print(f"stop_iter  = {res_idlg['stop_iter']}")
    print(f"pred_label = {res_idlg['pred_label']}")
    print(f"mses = {res_idlg['mses'][-1]}")
    log_result(res_idlg, "iDLG")

    # 同时展示两种方法最后的重建结果
    plt.figure("Comparison", figsize=(10, 5))

    if res_dlg["history"]:
        plt.subplot(1, 2, 1)
        plt.imshow(res_dlg["history"][-1])
        plt.title(f"DLG\nloss={res_dlg['final_loss']:.4f}")
        plt.axis("off")

    if res_idlg["history"]:
        plt.subplot(1, 2, 2)
        plt.imshow(res_idlg["history"][-1])
        plt.title(f"iDLG\nloss={res_idlg['final_loss']:.4f}")
        plt.axis("off")
    if args.image:
        IterPrint(res_dlg["history"], args.image + "_DLG")
        IterPrint(res_idlg["history"], args.image + "_iDLG")
    else:
        IterPrint(res_dlg["history"], f"cifar_{ args.index }_DLG")
        IterPrint(res_idlg["history"], f"cifar_{ args.index }_iDLG")


elif args.method:
    if args.method == "DLG":
        res = run_DLG()
    elif args.method == "iDLG":
        res = run_iDLG()

    log_result(res, args.method)
    history = res["history"]

    if history:
        IterPrint(history, args.method)

    else:
        print("No snapshots recorded; nothing to plot.")

    # show the last leaked image
    plt.figure(" Leaked images")
    plt.imshow(history[-1])
    plt.axis("off")

plt.show()
