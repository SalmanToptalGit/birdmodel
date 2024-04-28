from audiomentations import Compose

from utils.init_utils import init_logger, set_seed
from utils.data_utils import prepare_df_secondary_labels, prepare_df_config, prepare_df_year, prepare_year_data_split
import torch
import pandas as pd
from utils.augmentations import (CustomCompose, CustomOneOf, NoiseInjection, GaussianNoiseSNR, PinkNoiseSNR, BackgroundNoice)
import os
import numpy as np
from utils.metrics import calculate_metrics, metrics_to_string, calculate_competition_metrics
from nn.dataset import  BirdDataset
from nn.cnn import CNN, BCEKDLoss
from torch.utils.data.sampler import WeightedRandomSampler
from timm.scheduler import CosineLRScheduler
from utils.init_utils import AverageMeter
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import gc
from albumentations import HorizontalFlip,  Compose, Normalize, CoarseDropout
from albumentations.pytorch import ToTensorV2
import warnings
import copy
warnings.filterwarnings("ignore")


class Trainer:

    def prepare_dataframe_config(self, CFG):
        config = copy.deepcopy(CFG)

        os.makedirs(config["output_folder"], exist_ok=True)
        exp_folder = os.path.join(config["output_folder"], config["exp_name"])
        os.makedirs(exp_folder, exist_ok=True)

        set_seed(config["data_config"]["seed"])
        base_data_path = config["base_data_path"]

        df = pd.read_csv(os.path.join(base_data_path, 'birdclef-2024/train_metadata.csv'))
        config["data_config"] = prepare_df_config(df, config["data_config"])
        df = prepare_df_secondary_labels(df)
        df["path"] = os.path.join(base_data_path, "birdclef-2024/train_audio") + "/" + df["filename"]

        # Add Pseudo Labels Array
        pseudo_df = pd.read_csv(os.path.join(base_data_path, 'birdclef-2024/train_meta_data_pseudo.csv'))
        df['teacher_preds'] = pseudo_df[pseudo_df.columns.tolist()[2:]].values.tolist()

        df = prepare_year_data_split(df, config["data_config"]["kfold"], config["data_config"]["seed"])
        df["rating"] = np.clip(df["rating"] / df["rating"].max(), 0.1, 1.0)

        return df, config

    def batch_to_device(self, batch, device):
        batch_dict = {key: batch[key].to(device) for key in batch}
        return batch_dict


    def train_fn(self, data_loader, model, criterion, optimizer, scheduler, epoch, device, apex,
                                 max_grad_norm, target_columns):

        model.train()
        losses = AverageMeter()
        optimizer.zero_grad(set_to_none=True)
        scaler = GradScaler(enabled=apex)
        iters = len(data_loader)
        gt = []
        preds = []
        with tqdm(enumerate(data_loader), total=len(data_loader)) as t:
            for i, (batch) in t:
                batch = self.batch_to_device(batch, device)

                with autocast(enabled=apex):
                    outputs = model(batch)
                    loss = criterion(outputs)

                losses.update(loss.item(), batch["spec"].size(0))
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step(epoch + i / iters)
                t.set_postfix(
                    loss=losses.avg,
                    grad=grad_norm.item(),
                    lr=optimizer.param_groups[0]["lr"]
                )
                gt.append(batch["primary_targets"].cpu().detach().numpy())
                preds.append(outputs["logit"].sigmoid().cpu().detach().numpy())

        gt = np.concatenate(gt)
        preds = np.concatenate(preds)
        scores = calculate_competition_metrics(gt, preds, target_columns)
        return scores, losses.avg


    def val_fn(self, data_loader, model, criterion, device, apex, target_columns):
        model.eval()
        losses = AverageMeter()
        gt = []
        preds = []

        with tqdm(enumerate(data_loader), total=len(data_loader)) as t:
            for i, (batch) in t:
                batch = self.batch_to_device(batch, device)
                with autocast(enabled=apex):
                    with torch.no_grad():
                        outputs = model(batch["spec"])
                        batch["logit"] = outputs
                        loss = criterion(batch)

                losses.update(loss.item(), batch["spec"].size(0))
                t.set_postfix(loss=losses.avg)

                gt.append(batch["primary_targets"].cpu().detach().numpy())
                preds.append(batch["logit"].sigmoid().cpu().detach().numpy())

        gt = np.concatenate(gt)
        preds = np.concatenate(preds)
        scores = calculate_competition_metrics(gt, preds, target_columns)
        return scores, losses.avg

    def train_fold(self, df, config, fold):

        exp_folder = os.path.join(config["output_folder"], config["exp_name"])
        base_data_path = config["base_data_path"]
        data_config = config["data_config"]
        log_file = os.path.join(exp_folder, f"{fold}.log")
        logger = init_logger(log_file=log_file)

        logger.info("=" * 90)
        logger.info(f"Fold {fold} Training")
        logger.info("=" * 90)

        trn_df = df[df['fold'] != fold].reset_index(drop=True)
        val_df = df[df['fold'] == fold].reset_index(drop=True)
        print(trn_df.shape)
        logger.info(trn_df.shape)
        logger.info(trn_df['primary_label'].value_counts())
        logger.info(val_df.shape)
        logger.info(val_df['primary_label'].value_counts())
        labels = trn_df["primary_label"].values
        un_labels = np.unique(labels)
        weight = {t: 1.0 / len(np.where(labels == t)[0]) for t in un_labels}
        samples_weight = np.array([weight[t] for t in labels])
        sampler = WeightedRandomSampler(torch.from_numpy(samples_weight).type('torch.DoubleTensor'),
                                        len(samples_weight))

        train_transforms = Compose([
            HorizontalFlip(p=0.5),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
            CoarseDropout(p=0.5),
            ToTensorV2(p=1.0),
        ], p=1.)

        valid_transforms = Compose([
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
            ToTensorV2(p=1.0),
        ], p=1.)




        trn_dataset = BirdDataset(df=trn_df, config=data_config, wave_transforms=[], image_transform=train_transforms, is_train=True)
        train_loader = torch.utils.data.DataLoader(trn_dataset, shuffle=False, sampler=sampler,
                                                   **config["train_loader_config"])

        v_ds = BirdDataset(df=val_df.reset_index(drop=True), config=data_config, wave_transforms=[], image_transform=valid_transforms, is_train=False)
        val_loader = torch.utils.data.DataLoader(v_ds, shuffle=False, **config["val_loader_config"])

        model = CNN(config).to(config["device"])

        criterion = BCEKDLoss(num_classes=config["data_config"]["num_classes"])

        # optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=config["epochs"], T_mult=1, eta_min=1e-6, last_epoch=-1)

        optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr_max"], betas=(0.9, 0.999), eps=1e-08,
                                      weight_decay=config["weight_decay"], amsgrad=False, )
        scheduler = CosineLRScheduler(optimizer, t_initial=10, warmup_t=1, cycle_limit=40, cycle_decay=1.0,
                                      lr_min=config["lr_min"], t_in_epochs=True, )

        patience = config["early_stopping"]
        best_score = 0.0
        n_patience = 0

        for epoch in range(1, config["epochs"] + 1):

            train_scores, train_losses_avg = self.train_fn(data_loader=train_loader, model=model,
                                                      criterion=criterion, optimizer=optimizer,
                                                      scheduler=scheduler,
                                                      epoch=0, device=config["device"],
                                                      apex=config["apex"],
                                                      max_grad_norm=config["max_grad_norm"],
                                                      target_columns=config["data_config"]["target_columns"])

            train_scores_str = metrics_to_string(train_scores, "Train")
            train_info = f"Epoch {epoch} - Train loss: {train_losses_avg:.4f}, {train_scores_str}"
            logger.info(train_info)

            val_scores, val_losses_avg = self.val_fn(data_loader=val_loader, model=model, criterion=criterion,
                                                device=config["device"],
                                                apex=config["apex"], target_columns=config["data_config"]["target_columns"])

            val_scores_str = metrics_to_string(val_scores, f"Valid")
            val_info = f"Epoch {epoch} - Valid loss: {val_losses_avg:.4f}, {val_scores_str}"
            logger.info(val_info)

            val_score = val_scores["cmAP_1"]

            is_better = val_score > best_score
            best_score = max(val_score, best_score)

            exp_name = config["exp_name"]

            if is_better:
                state = {
                    "epoch": epoch,
                    "state_dict": model.state_dict(),
                    "best_loss": best_score,
                    "optimizer": optimizer.state_dict(),
                }
                logger.info(
                    f"Epoch {epoch} - Save Best Score: {best_score:.4f} Model\n")
                torch.save(
                    state,
                    os.path.join(exp_folder, f"{fold}.bin")
                )
                n_patience = 0
            else:
                n_patience += 1
                logger.info(
                    f"Valid loss didn't improve last {n_patience} epochs.\n")

            if n_patience >= patience:
                logger.info(
                    "Early stop, Training End.\n")
                break

    def train_folds(self, config):
        df, config = self.prepare_dataframe_config(config)
        for fold in config["folds"]:
            self.train_fold(df, config, fold)
