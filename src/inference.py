import concurrent.futures
import glob
import multiprocessing
import os
import shutil

import albumentations
import librosa
import numpy as np
import pandas as pd
import pytorch_lightning as L
import timm
import torch
import torchaudio
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm.notebook import tqdm


class cfg:
    data_path = "../data"
    data_path = "/kaggle/input/birdclef-2024"

    normalize_waveform = True
    sample_rate = 32000
    n_fft = 2048
    hop_length = 512
    window_length = None
    melspec_hres = 128
    melspec_wres = 312
    freq_min = 40
    freq_max = 15000
    log_scale_power = 2
    max_decibels = 100
    frame_duration = 5
    frame_rate = sample_rate / hop_length

    # vit_b0 = "efficientvit_b0.r224_in1k"
    vit_b1 = "efficientvit_b1.r224_in1k"
    # vit_b1 = "efficientvit_b1.r288_in1"
    # effnet_b0 = "tf_efficientnetv2_b0.in1k"
    # effnet_b1 = "tf_efficientnetv2_b1.in1k"
    # vit_m0 = "efficientvit_m0.r224_in1k"
    # vit_m1 = "efficientvit_m1.r224_in1k"
    backbone = vit_b1

    accelerator = "cpu"
    precision = "bf16-mixed"
    n_workers = multiprocessing.cpu_count()


class TestDataset(Dataset):
    def __init__(self, df: pd.DataFrame, waveform: np.ndarray, augmentation):

        self.df = df
        self.waveform = waveform
        self.augmentation = augmentation

        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=cfg.sample_rate,
            n_mels=cfg.melspec_hres,
            f_min=cfg.freq_min,
            f_max=cfg.freq_max,
            n_fft=cfg.n_fft,
            hop_length=cfg.hop_length,
            normalized=cfg.normalize_waveform,
            center=True,
            pad_mode="reflect",
            norm="slaney",
            mel_scale="slaney",
        )
        self.db_transform = torchaudio.transforms.AmplitudeToDB(
            stype="power", top_db=cfg.max_decibels
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        sample = self.df.loc[idx, :]
        row_id = sample.row_id

        end = int(sample.seconds)
        start = int(end - 5)

        waveform = self.waveform[cfg.sample_rate * start : cfg.sample_rate * end]
        waveform = torch.tensor(waveform, dtype=torch.float32)
        melspec = self.db_transform(self.mel_transform(waveform)).type(torch.uint8)
        melspec = melspec.expand(3, -1, -1).permute(1, 2, 0).numpy()

        if self.augmentation is not None:
            melspec = self.augmentation(image=melspec)["image"]

        return {
            "row_id": row_id,
            "melspec": melspec,
        }


class EfficientViT(L.LightningModule):
    def __init__(self):
        super().__init__()

        self.vit = timm.create_model(
            cfg.backbone,
            pretrained=False,
            num_classes=182,
        )

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        out = self.vit(x)

        return out


def predict(filepath: str) -> dict:
    prediction_dict = {}
    waveform, _ = librosa.load(filepath, sr=cfg.sample_rate)

    name_ = filepath.split(".ogg")[0].split("/")[-1]
    row_ids = [name_ + f"_{second}" for second in seconds]

    test_df = pd.DataFrame(
        {
            "row_id": row_ids,
            "seconds": seconds,
        }
    )

    dataset = TestDataset(df=test_df, waveform=waveform, augmentation=test_augmentation)

    test_loader = DataLoader(
        dataset,
        batch_size=4,
        num_workers=os.cpu_count(),
        drop_last=False,
        shuffle=False,
        pin_memory=True,
    )

    for batch in test_loader:
        row_ids = batch["row_id"]
        batch.pop("row_id")

        for row_id in row_ids:
            if row_id not in prediction_dict:
                prediction_dict[str(row_id)] = []

        with torch.no_grad():
            output = model(batch["melspec"])

        for row_id_idx, row_id in enumerate(row_ids):
            prediction_dict[str(row_id)].append(
                output[row_id_idx, :].sigmoid().detach().numpy()
            )

    for row_id in list(prediction_dict.keys()):
        logits = prediction_dict[row_id]
        logits = np.array(logits)[0]
        prediction_dict[row_id] = {}
        for label in range(len(target_columns)):
            prediction_dict[row_id][target_columns[label]] = logits[label]

    return prediction_dict


if __name__ == "__main__":
    submission_df = pd.read_csv("../data/sample_submission.csv")
    # submission_df = pd.read_csv("../input/birdclef-2024/sample_submission.csv")
    target_columns_ = submission_df.columns.tolist()
    target_columns = submission_df.columns.tolist()[1:]

    chunks = 48
    test_path = "../data/test_soundscapes/"
    files = glob.glob(f"{test_path}*")
    # if len(files) == 1:
    chunks = 2

    seconds = [i for i in range(5, (chunks * 5) + 5, 5)]

    test_path = "../data/test_soundscapes/"
    # test_path = "../data/test_soundscapes/"

    files = glob.glob(f"{test_path}*")
    if len(files) == 1:
        test_path = "../data/test_soundscapes/"
    print(test_path)

    trainer = L.Trainer(accelerator="cpu", precision="bf16-mixed")
    with trainer.init_module():
        model = EfficientViT.load_from_checkpoint(
            "../model_objects/2024-05-26 21_34_51_efficientvit_b1.r224_in1k_foundation_val_0.33_lr_0.0001_decay_1e-06_fold_0_epochs_50.ckpt",
            map_location=torch.device("cpu"),
        )
    model._trainer = trainer
    model = torch.jit.optimize_for_inference(torch.jit.script(model.eval()))

    test_filepaths = list(glob.glob(f"{test_path}*.ogg"))
    test_augmentation = albumentations.Compose([albumentations.Normalize(p=1)])

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        dicts = list(
            tqdm(executor.map(predict, test_filepaths), total=len(test_filepaths))
        )

    prediction_dicts = {}
    for d in dicts:
        prediction_dicts.update(d)

    submission = (
        pd.DataFrame.from_dict(prediction_dicts, "index")
        .rename_axis("row_id")
        .reset_index()
    )
    submission.to_csv("submission.csv", index=False)
    print("Done")


submission
