
import torch.utils.data as torchdata
import librosa
import torch
import numpy as np
import warnings
warnings.filterwarnings("ignore")



class BirdDataset(torchdata.Dataset):

    def __init__(self, df, config, num_classes, add_secondary_labels=False):
        self.df = df
        self.bird2id = config['bird2id']
        self.period = config['period']
        self.secondary_coef = config['secondary_coef']
        self.df["secondary_labels"] = (
            self.df["secondary_labels"]
            .map(
                lambda s: s.replace("[", "")
                .replace("]", "")
                .replace(",", "")
                .replace("'", "")
                .split(" ")
            ).values
        )

        self.smooth_label = config['smooth_label']
        self.num_classes = num_classes
        self.add_secondary_labels = add_secondary_labels

    def __len__(self):
        return len(self.df)

    def prepare_target(self, idx):
        target = np.zeros(self.num_classes, dtype=np.float32)
        if self.df["primary_label"].iloc[idx] != 'nocall':
            primary_label = self.bird2id[self.df["primary_label"].iloc[idx]]
            target[primary_label] = 1.0
            if self.add_secondary_labels:
                for s in self.df["secondary_labels"].iloc[idx]:
                    if s != "" and s in self.bird2id.keys():
                        target[self.bird2id[s]] = self.secondary_coef
        target = torch.from_numpy(target).float()
        return target

    def load_wave_and_crop(self, filename, period, start=None):

        waveform_orig, sample_rate = librosa.load(filename, sr=32000, mono=False)

        wave_len = len(waveform_orig)
        waveform = np.concatenate([waveform_orig, waveform_orig, waveform_orig])

        effective_length = sample_rate * period
        while len(waveform) < (period * sample_rate * 3):
            waveform = np.concatenate([waveform, waveform_orig])
        if start is not None:
            start = start - (period - 5) / 2 * sample_rate
            while start < 0:
                start += wave_len
            start = int(start)
        else:
            if wave_len < effective_length:
                start = np.random.randint(effective_length - wave_len)
            elif wave_len > effective_length:
                start = np.random.randint(wave_len - effective_length)
            elif wave_len == effective_length:
                start = 0

        waveform_seg = waveform[start: start + int(effective_length)]

        return waveform_orig, waveform_seg, sample_rate, start

    def __getitem__(self, idx):
        path = self.df["path"].iloc[idx]

        waveform_orig, waveform_seg, sample_rate, start = self.load_wave_and_crop(path, period=self.period, start=0)
        waveform_seg = torch.from_numpy(np.nan_to_num(waveform_seg)).float()
        rating = self.df["rating"].iloc[idx]
        target = self.prepare_target(idx)

        batch_dict = {
            "wave": waveform_seg,
            "rating": rating,
            "primary_targets": (target > 0.5).float(),
            "smooth_targets": target * (1 - self.smooth_label) + self.smooth_label / target.size(-1),
        }

        return batch_dict
