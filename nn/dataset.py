from utils.data_utils import load_and_crop_range
import numpy as np
import torch
import torchaudio
import torch.utils.data as torchdata
class BirdDataset(torchdata.Dataset):

    def __init__(self, df, config, wave_transforms=[], image_transform = None, is_train=False):
        self.df = df
        self.bird2id = config['bird2id']
        self.period = config['period']
        self.wave_transforms = wave_transforms
        self.num_classes = config['num_classes']
        self.is_train = is_train
        self.secondary_coef = config['secondary_coef']
        self.smooth_label = config['smooth_label']
        self.image_transform = image_transform
        mel_spec_params = config["mel_spec_paramms"]
        self.mel_spec_extractor = torchaudio.transforms.MelSpectrogram(sample_rate=32000,
                                                                  hop_length=mel_spec_params["hop_length"],
                                                                  n_mels=mel_spec_params["n_mels"],
                                                                  f_min=mel_spec_params["f_min"],
                                                                  f_max=mel_spec_params["f_max"],
                                                                  n_fft=mel_spec_params["n_fft"],
                                                                  center=True, pad_mode='constant', norm='slaney',
                                                                  onesided=True, mel_scale='slaney')
        self.db_transform = torchaudio.transforms.AmplitudeToDB(stype='power', top_db=mel_spec_params["top_db"])


    def __len__(self):
        return len(self.df)

    def normalize(self, x):
        eps = 1e-6
        mean = x.mean((1, 2), keepdim=True)
        std = x.std((1, 2), keepdim=True)
        xstd = (x - mean) / (std + eps)
        norm_max = torch.amax(xstd, dim=(1, 2), keepdim=True)
        norm_min = torch.amin(xstd, dim=(1, 2), keepdim=True)
        x = (xstd - norm_min) / (norm_max - norm_min + eps)
        return x

    def wave_to_spec(self, x):
        x = self.normalize(self.db_transform(self.mel_spec_extractor(x)).unsqueeze(0))[0, ]
        x = x.unsqueeze(2).expand(-1, -1, 3)
        x = (x * 255).numpy()
        if self.image_transform:
            x = self.image_transform(image=x)['image']
        return x



    def get_item(self, idx):
        path = self.df["path"].iloc[idx]

        wave = load_and_crop_range(path, period=self.period, resample=True, sr=32000, normalize=False, start=0)
        spec = self.wave_to_spec(torch.from_numpy(wave).float())

        # if len(self.wave_transforms) > 0:
        #     for i in range(0, len(self.wave_transforms)):
        #         wave = self.wave_transforms[i](wave)

        target = np.zeros(self.num_classes, dtype=np.float32)
        primary_label = self.bird2id[self.df["primary_label"].iloc[idx]]
        target[primary_label] = 1.0

        if self.is_train:
            for s in self.df["primary_label"].iloc[idx]:
                if s != "" and s in self.bird2id.keys():
                    target[self.bird2id[s]] = self.secondary_coef


        pseudo_ = self.df["teacher_preds"].iloc[idx]
        teacher_preds = torch.from_numpy(np.nan_to_num(pseudo_)).float()
        target = torch.from_numpy(target).float()
        return {
            "spec": spec,
            "rating": torch.tensor(self.df["rating"].iloc[idx]),
            "primary_targets": (target > 0.5).float(),
            "loss_target": target * (1 - self.smooth_label) + self.smooth_label / target.size(-1),
            "teacher_preds": teacher_preds,
        }

    def __getitem__(self, idx):

        idx_dict = self.get_item(idx)

        return idx_dict
