from audiomentations import Compose

from utils.init_utils import init_logger, set_seed
from utils.data_utils import prepare_df_secondary_labels, prepare_df_config, prepare_df_year, prepare_year_data_split
import torch
import pandas as pd
import os
import numpy as np
from utils.metrics import calculate_metrics, metrics_to_string, calculate_competition_metrics
from nn.dataset import  BirdDataset
from nn.cnn import CNN
from nn.losses import *
from torch.utils.data.sampler import WeightedRandomSampler
from timm.scheduler import CosineLRScheduler
from utils.init_utils import AverageMeter
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from nn.sampler import MultilabelBalancedRandomSampler
import warnings
import copy
warnings.filterwarnings("ignore")
from utils.data_utils import setup_output_dir, read_dataframe, normalize_rating, do_kfold, prepare_teacher_pred

class Trainer:

    def batch_to_device(self, batch, device):
        batch_dict = {key: batch[key].to(device) for key in batch}
        return batch_dict

    def train_one_epoch(self, data_loader, model, criterion, optimizer, scheduler, epoch, device, apex,
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

                losses.update(loss.item(), batch["wave"].size(0))
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                # grad_norm = torch.nn.utils.clip_grad_norm_(
                #     model.parameters(), max_norm=max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step(epoch + i / iters)
                t.set_postfix(
                    loss=losses.avg,
                    # grad=grad_norm.item(),
                    lr=optimizer.param_groups[0]["lr"]
                )
                gt.append(batch["primary_targets"].cpu().detach().numpy())
                preds.append(outputs["logit"].sigmoid().cpu().detach().numpy())

        gt = np.concatenate(gt)
        preds = np.concatenate(preds)
        scores = calculate_competition_metrics(gt, preds, target_columns)
        return scores, losses.avg

    def validate_one_epoch(self, data_loader, model, criterion, device, apex, target_columns):
        model.eval()
        losses = AverageMeter()
        gt = []
        preds = []

        with tqdm(enumerate(data_loader), total=len(data_loader)) as t:
            for i, (batch) in t:
                batch = self.batch_to_device(batch, device)
                with autocast(enabled=apex):
                    with torch.no_grad():
                        outputs = model(batch)
                        loss = criterion(outputs)

                losses.update(loss.item(), batch["wave"].size(0))
                t.set_postfix(loss=losses.avg)

                gt.append(batch["primary_targets"].cpu().detach().numpy())
                preds.append(batch["logit"].sigmoid().cpu().detach().numpy())

        gt = np.concatenate(gt)
        preds = np.concatenate(preds)
        scores = calculate_competition_metrics(gt, preds, target_columns)
        return scores, losses.avg

    def train_fold(self, df, config, fold):

        logger = init_logger(log_file=os.path.join(config["exp_folder"], f"{fold}.log"))

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

        # labels = trn_df["primary_label"].values
        # un_labels = np.unique(labels)
        # weight = {t: 1.0 / len(np.where(labels == t)[0]) for t in un_labels}
        # samples_weight = np.array([weight[t] for t in labels])
        # sampler = WeightedRandomSampler(torch.from_numpy(samples_weight).type('torch.DoubleTensor'),
        #                                 len(samples_weight))


        if config["sampler"]:
            one_hot_target = np.zeros((trn_df.shape[0], len(config["target_columns"])), dtype=np.float32)

            for i, label in enumerate(trn_df.primary_label):
                primary_label = config["bird2id"][label]
                one_hot_target[i, primary_label] = 1.0

            sampler = MultilabelBalancedRandomSampler(
                one_hot_target,
                trn_df.index,
                class_choice="least_sampled"
            )
            shuffle = False
        else:
            sampler = None
            shuffle = True


        trn_dataset = BirdDataset(df=trn_df.reset_index(drop=True), config=config,
                                  num_classes=len(config["target_columns"]),
                                  add_secondary_labels=True)
        train_loader = torch.utils.data.DataLoader(trn_dataset, shuffle=shuffle, sampler=sampler,
                                                   **config["train_loader_config"])

        v_ds = BirdDataset(df=val_df.reset_index(drop=True), config=config, num_classes=len(config["target_columns"]),
                           add_secondary_labels=True)
        val_loader = torch.utils.data.DataLoader(v_ds, shuffle=False, **config["val_loader_config"])

        model = CNN(config).to(config["device"])

        if config["KD"]:
            criterion = BCEKDLoss()
        else:
            criterion = FocalLossBCE()


        optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr_max"], betas=(0.9, 0.999), eps=1e-08,
                                      weight_decay=config["weight_decay"], amsgrad=False, )
        scheduler = CosineLRScheduler(optimizer, t_initial=10, warmup_t=1, cycle_limit=40, cycle_decay=1.0,
                                      lr_min=config["lr_min"], t_in_epochs=True, )

        patience = config["early_stopping"]
        best_score = 0.0
        n_patience = 0

        for epoch in range(1, config["epochs"] + 1):

            train_scores, train_losses_avg = self.train_one_epoch(data_loader=train_loader, model=model,
                                                             criterion=criterion, optimizer=optimizer,
                                                             scheduler=scheduler,
                                                             epoch=0, device=config["device"],
                                                             apex=config["apex"],
                                                             max_grad_norm=config["max_grad_norm"],
                                                             target_columns=config["target_columns"])

            train_scores_str = metrics_to_string(train_scores, "Train")
            train_info = f"Epoch {epoch} - Train loss: {train_losses_avg:.4f}, {train_scores_str}"
            logger.info(train_info)

            val_scores, val_losses_avg = self.validate_one_epoch(data_loader=val_loader, model=model, criterion=criterion,
                                                            device=config["device"],
                                                            apex=config["apex"],
                                                            target_columns=config["target_columns"])

            val_scores_str = metrics_to_string(val_scores, f"Valid")
            val_info = f"Epoch {epoch} - Valid loss: {val_losses_avg:.4f}, {val_scores_str}"
            logger.info(val_info)

            val_score = val_scores["ROC"]

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
                    os.path.join(config["exp_folder"], f"{fold}.bin")
                )
                n_patience = 0
            else:
                n_patience += 1
                logger.info(
                    f"Valid loss didn't improve last {n_patience} epochs.\n")

            if n_patience >= patience:
                logger.info(
                    "Early stop, Training End.\n")
                state = {
                    "epoch": epoch,
                    "state_dict": model.state_dict(),
                    "best_loss": best_score,
                    "optimizer": optimizer.state_dict(),
                }
                torch.save(
                    state,
                    os.path.join(config["exp_folder"], f"final_{fold}.bin")
                )
                break


    def train_folds(self, config):
        config = setup_output_dir(config)
        df = read_dataframe()
        df["teacher_preds"] = prepare_teacher_pred(config["target_columns"])
        set_seed(config["seed"])
        df["rating"] = normalize_rating(df)
        df = do_kfold(df, KFOLD=5, seed=config["seed"])
        for fold in config["fold"]:
            self.train_fold(df, config, fold)
