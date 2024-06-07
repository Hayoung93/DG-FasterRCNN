import os
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from config import get_cfg
from dataset import build, collate_fn
from model import FasterRCNN_noFPN


def main(cfg):
    # train variables
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    start_epoch = 0
    best_train_loss = torch.inf
    best_score = 0.0
    writer = SummaryWriter(os.path.join(cfg.io.save_dir, cfg.io.exp_name))
    # data
    trainset = build(cfg, "train", "daytime_clear_train.json")
    trainloader = DataLoader(trainset, cfg.run.batch_size, True, num_workers=cfg.run.num_workers, collate_fn=collate_fn)
    valset_source = build(cfg, "val", "daytime_clear_test.json")
    valloader_source = DataLoader(valset_source, 1, False, num_workers=cfg.run.num_workers)
    valset_targets = [build(cfg, "val", "night_sunny_test.json"), build(cfg, "val", "dusk_rainy_train.json"), build(cfg, "val", "night_rainy_train.json"), build(cfg, "val", "daytime_foggy_test.json")]
    valloader_targets = [DataLoader(vset, 1, False, num_workers=cfg.run.num_workers) for vset in valset_targets]
    # model
    model = FasterRCNN_noFPN(cfg)
    model.to(device)
    # load learned augs
    if len(cfg.domain.weather_domains) != 0 and len(cfg.domain.time_domains) != 0:
        assert cfg.io.learned_aug != "" and os.path.isfile(cfg.io.learned_aug), "No learned aug found from: {} | Weathers: {} | Times: {}".format(cfg.io.learned_aug, cfg.domain.weather_domains, cfg.domain.time_domains)
        cp_aug = torch.load(cfg.io.learned_aug)
        dict_aug = {}
        for cp_k, cp_v in cp_aug["model"].items():
            if ("style_params" in cp_k) or ("semantic_aug" in cp_k):
                dict_aug[cp_k] = cp_v
        msg = model.load_state_dict(dict_aug, strict=False)
    # optmization
    param_dicts = [
        {"params": [p for n, p in model.named_parameters() if not("backbone" in n) and p.requires_grad], "lr": cfg.run.optim.lr},
        {"params": [p for n, p in model.named_parameters() if "backbone" in n], "lr": cfg.run.optim.lr_backbone, "weight_decay": cfg.run.optim.lr_backbone * 0.1}
    ]
    optimizer = eval("torch.optim.{}".format(cfg.run.optim.optimizer))(param_dicts, lr=cfg.run.optim.lr, weight_decay=cfg.run.optim.weight_decay)
    scheduler = eval("torch.optim.lr_scheduler.{}".format(cfg.run.optim.scheduler))(optimizer, cfg.run.epochs)
    # resume
    if (cfg.io.resume is not None) and (os.path.isfile(cfg.io.resume)):
        cp = torch.load(cfg.io.resume, map_location=device)
        new_cp = {}
        for cp_k, cp_v in cp["model"].items():
            new_cp[cp_k.replace("module.", "")] = cp_v
        msg = model.load_state_dict(new_cp)
        print("Loaded checkpoint from {}".format(cfg.io.resume))
        optimizer.load_state_dict(cp["optimizer"])
        scheduler.load_state_dict(cp["scheduler"])
        start_epoch = cp["epoch"] + 1
        best_train_loss = cp["best_train_loss"]
        best_score = cp["best_score"]
    # train
    if cfg.run.parallel == "DP":
        model = torch.nn.DataParallel(model)
    elif cfg.run.parallel == "DDP":
        raise NotImplementedError
    model.train()
    pbar_epoch = tqdm(range(start_epoch, cfg.run.epochs), position=1)
    for ep in pbar_epoch:
        pbar_epoch.set_description("Epoch: {}".format(ep))
        total_loss = 0.0
        pbar_iter_train = tqdm(enumerate(trainloader), position=2, total=len(trainloader))
        for it, data in pbar_iter_train:
            inputs, labels = data
            inputs = inputs.cuda()
            loss_dict = model(inputs, labels)
            losses = 0.0
            description = ["Epoch: {}".format(ep), "Iter: {}/{}".format(it, len(trainloader))]
            for k, l in loss_dict.items():
                losses = losses + l
                description.append("{}: {:.4f}".format(k, l))
            description = " | ".join(description)
            running_loss = losses.item()
            total_loss += running_loss
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            pbar_iter_train.set_description(description)
        if total_loss < best_train_loss:
            best_train_loss = total_loss
        writer.add_scalar("Loss", total_loss, ep)
        pbar_epoch.set_description("Epoch: {} | Total Loss: {}".format(ep, total_loss))
        scheduler.step()
        # validation using source test data
        if (ep + 1) % cfg.run.val_interval == 0:
            model.eval()
            pbar_iter_val = tqdm(enumerate(valloader), position=3)
            with torch.inference_mode():
                tps, fps, fns = 0, 0, 0
                dices = []
                for it, data in pbar_iter_val:
                    inputs, labels, _ = data
                    if cfg.model.dimension == 2:
                        outputs = []
                        for ci in range(inputs.shape[2] - cfg.model.in_channels + 1):
                            _outputs = model(inputs[:, 0, ci:ci+cfg.model.in_channels, ...])
                            outputs.append(_outputs)
                            pbar_iter_train.set_description("Iter: {}/{} | SubIndex: {}/{}".format(it, len(valloader), ci, inputs.shape[2] - cfg.model.in_channels))
                        outputs = torch.stack(outputs, dim=2)
                    elif cfg.model.dimension == 3:
                        outputs = model(inputs)
                    tp, fp, fn = criterion.compute_conf(outputs, labels.to(torch.int64)[:, :, 1:-1, ...], slice_wise=False)
                    tp, fp, fn = tp[1], fp[1], fn[1]
                    tps += tp; fps += fp; fns += fn
                    dice = (2 * tp + 1e-5) / (2 * tp + fp + fn + 1e-5)
                    dices.append(dice)
                    pbar_iter_val.set_description("Iter: {}/{} | Dice: {:.4f} | TP: {:.4f} | FP: {:.4f} | FN: {:.4f}".format(it, len(valloader), dice, tp / labels.numel(), fp / labels.numel(), fn / labels.numel()))
                n, h, w = cfg.transform.target_size
                denom = n * h * w
                tps, fps, fns = tps / denom, fps / denom, fns / denom
                dices = torch.tensor(dices)
                val_avg_dice = dices.mean()
                writer.add_scalar("Metric/val_avg_dice", val_avg_dice, ep)
                writer.add_scalar("Metric/TP", tps, ep)
                writer.add_scalar("Metric/FP", fps, ep)
                writer.add_scalar("Metric/FN", fns, ep)
                if val_avg_dice > best_score:
                    best_score = val_avg_dice
                    torch.save({
                        "model": model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "epoch": ep,
                        "best_train_loss": best_train_loss,
                        "best_score": best_score
                    }, os.path.join(cfg.io.save_dir, cfg.io.exp_name, "checkpoint_best.pth"))
        torch.save({
            "model": model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": ep,
            "best_train_loss": best_train_loss,
            "best_score": best_score
        }, os.path.join(cfg.io.save_dir, cfg.io.exp_name, "checkpoint.pth"))
    # last validation using all target data
    ...
    torch.save({
        "model": model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": ep,
        "best_train_loss": best_train_loss,
        "best_score": best_score
    }, os.path.join(cfg.io.save_dir, cfg.io.exp_name, "checkpoint_last.pth"))


if __name__ == "__main__":
    cfg = get_cfg()
    os.makedirs(os.path.join(cfg.io.save_dir, cfg.io.exp_name), exist_ok=True)
    cfg.domain.weather_domains = [wd for wd in cfg.domain.weather_domains if wd != ""]
    cfg.domain.time_domains = [wd for wd in cfg.domain.time_domains if wd != ""]
    main(cfg)
