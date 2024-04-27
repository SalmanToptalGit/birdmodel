from utils.data_utils import load_and_crop_range
import numpy as np
import torch
import torch.utils.data as torchdata
class BirdDataset(torchdata.Dataset):

    def __init__(self, df, config, wave_transforms=[], is_train=False):
        self.df = df
        self.bird2id = config['bird2id']
        self.period = config['period']
        self.wave_transforms = wave_transforms
        self.num_classes = config['num_classes']
        self.is_train = is_train
        self.secondary_coef = config['secondary_coef']
        self.smooth_label = config['smooth_label']

    def __len__(self):
        return len(self.df)

    def get_item(self, idx):
        path = self.df["path"].iloc[idx]

        wave = load_and_crop_range(path, period=self.period, resample=True, sr=32000, normalize=True, start=0)
        if len(self.wave_transforms) > 0:
            for i in range(0, len(self.wave_transforms)):
                wave = self.wave_transforms[i](wave)

        target = np.zeros(self.num_classes, dtype=np.float32)
        primary_label = self.bird2id[self.df["primary_label"].iloc[idx]]
        target[primary_label] = 1.0

        if self.is_train:
            for s in self.df["primary_label"].iloc[idx]:
                if s != "" and s in self.bird2id.keys():
                    target[self.bird2id[s]] = self.secondary_coef


        pseudo_ = self.df["teacher_preds"].iloc[idx]
        teacher_preds = torch.from_numpy(np.nan_to_num(pseudo_)).float()
        wave = torch.from_numpy(np.nan_to_num(wave)).float()
        target = torch.from_numpy(target).float()
        return {
            "wave": wave,
            "rating": self.df["rating"].iloc[idx],
            "primary_targets": (target > 0.5).float(),
            "loss_target": target * (1 - self.smooth_label) + self.smooth_label / target.size(-1),
            "teacher_preds": teacher_preds,
        }

    def __getitem__(self, idx):

        idx_dict = self.get_item(idx)

        return idx_dict
