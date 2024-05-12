import gc
import itertools
import logging
import pickle

import librosa
import numpy as np
import pandas as pd
import scipy
import seaborn as sns
import soundfile
import torch
import torchaudio
from filenames_to_drop import filenames_to_drop
from IPython.display import Audio
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm.notebook import tqdm

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format=" %(asctime)s [%(threadName)s] [%(levelname)s] 🐦‍🔥 %(message)s",
)

sns.set(
    rc={
        "figure.figsize": (8, 3),
        "figure.dpi": 240,
    }
)
sns.set_style("darkgrid", {"axes.grid": False})
sns.set_context("paper", font_scale=0.6)


class cfg:
    root_folder = "input/birdclef-2024"
    normalize_waveform = True
    sample_rate = 32000
    n_fft = 2048
    hop_length = 512
    window_length = None
    melspec_hres = 128
    melspec_wres = 312
    freq_min = 20
    freq_max = 16000
    log_scale_power = 3
    create_frames = True
    frame_duration = None
    max_decibels = 80
    frame_rate = sample_rate / hop_length


def load_data(path: str = cfg.root_folder) -> pd.DataFrame:
    logging.info("Loading data")
    model_input_df = pd.read_csv(
        f"{cfg.root_folder}/train_metadata.csv",
        dtype={
            "secondary_labels": "string",
            "primary_label": "category",
        },
    )

    submission_df = pd.read_csv(f"{cfg.root_folder}/sample_submission.csv")
    taxonomy_df = pd.read_csv(f"{cfg.root_folder}/eBird_Taxonomy_v2021.csv")

    return model_input_df, submission_df, taxonomy_df


def remove_samples(df: pd.DataFrame, filenames_to_drop: list[str]) -> pd.DataFrame:
    logging.info("Removing hand-selected samples from data")
    df = df[~df["filename"].isin(filenames_to_drop)]
    df = df.reset_index(drop=True)
    df = df.reset_index(drop=False, names="sample_index")
    return df


def create_label_mappings(submission_df: pd.DataFrame) -> dict:
    logging.info("Creating label mappings")
    cfg.labels = submission_df.columns[1:]
    cfg.n_classes = len(cfg.labels)

    class_to_label_map = dict(zip(cfg.labels, np.arange(cfg.n_classes)))
    label_to_class_map = dict([(v, k) for k, v in class_to_label_map.items()])

    return class_to_label_map, label_to_class_map


def create_log_melspec(
    path: str,
    normalize: bool = cfg.normalize_waveform,
) -> np.ndarray:
    waveform, _ = librosa.load(path, sr=cfg.sample_rate, duration=cfg.frame_duration)

    if normalize:
        waveform = librosa.util.normalize(waveform)

    melspec = librosa.feature.melspectrogram(
        y=waveform,
        sr=cfg.sample_rate,
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_length,
        n_mels=cfg.melspec_hres,
        fmin=cfg.freq_min,
        fmax=cfg.freq_max,
        power=cfg.log_scale_power,
    )

    melspec = librosa.power_to_db(melspec, ref=cfg.max_decibels)
    melspec = melspec - melspec.min()
    melspec = (melspec / melspec.max() * 255).astype(np.uint8)

    return melspec, waveform


def visualize_train_activity_detection(
    model_input_df: pd.DataFrame, idx: int = None
) -> None:
    if idx is None:
        logging.info("Selecting random sample for data review")
        idx = np.random.randint(low=0, high=len(model_input_df))

    path = f"{cfg.root_folder}/train_audio/{model_input_df.loc[idx]['filename']}"

    melspec, waveform = create_log_melspec(path=path)
    energy = melspec.sum(axis=0)
    peaks = scipy.signal.find_peaks_cwt(
        energy,
        widths=np.arange(20, 25),
        wavelet=None,
        max_distances=None,
        gap_thresh=None,
        min_length=1,
        min_snr=1.5,
        noise_perc=15,
        window_size=None,
    )

    top_peaks = np.argsort(-energy[peaks])[:5]
    top_peaks_index = peaks[top_peaks]

    selected_windows = []
    for i in top_peaks_index:
        selected_windows.append(
            [i - (cfg.frame_rate * 2.5), i + (cfg.frame_rate * 2.5)]
        )

    fig, ax = plt.subplots(2, 1, sharex=True)
    sns.heatmap(melspec, ax=ax[0], cbar=False)

    for window in selected_windows:
        ax[1].axvspan(window[0], window[1], color="#F7C566", alpha=0.3)

    sns.lineplot(energy, ax=ax[1], linewidth=0.5, color="#ef476f")
    sns.scatterplot(x=peaks, y=energy[peaks], markers=".", color="#6C0345", ax=ax[1])

    ax[0].yaxis.set_visible(False)
    ax[1].yaxis.set_visible(False)
    ax[1].xaxis.set_visible(False)
    fig.suptitle(
        f"index: {idx}  |  "
        + f"label: {model_input_df.loc[idx]['primary_label']}  |  "
        + f"name: {model_input_df.loc[idx]['common_name']}  |  "
        + f"type: {model_input_df.loc[idx]['type']}  |  "
        + f"secondary: {model_input_df.loc[idx]['secondary_labels']}\n"
        + f"shape: {melspec.shape}  |  "
        + f"duration: {len(waveform) / cfg.sample_rate:.4}s  |  "
        + f"min: {melspec.min():.0f}, max: {melspec.max():.0f}  |  "
        + f"µ: {melspec.mean():.1f}, σ: {melspec.std():.1f}",
    )
    plt.tight_layout()
    plt.show()

    sns.lineplot(waveform, linewidth=0.5, color="#ef476f").yaxis.set_visible(False)
    for window in selected_windows:
        if window[0] < 0:
            window[0] = 0
        start = window[0] / cfg.frame_rate * cfg.sample_rate
        end = window[1] / cfg.frame_rate * cfg.sample_rate
        plt.axvspan(start, end, color="#F7C566", alpha=0.3)
    plt.tight_layout()
    plt.show()


def detect_train_activity_windows(melspec: np.ndarray) -> list:
    energy_per_frame = melspec.sum(axis=0)
    peak_idx = scipy.signal.find_peaks_cwt(
        energy_per_frame,
        widths=np.arange(20, 25),
        wavelet=None,
        max_distances=None,
        gap_thresh=None,
        min_length=1,
        min_snr=1.5,
        noise_perc=15,
        window_size=None,
    )

    if peak_idx.size == 0:  # if no peaks are detected take the first 5 seconds
        selected_windows = [[0, cfg.frame_rate * 5]]
    else:
        top_peaks = np.argsort(-energy_per_frame[peak_idx])[:5]
        top_peaks_idx = peak_idx[top_peaks]

        selected_windows = []
        for peak in top_peaks_idx:
            selected_windows.append(
                [peak - (cfg.frame_rate * 2.5), peak + (cfg.frame_rate * 2.5)]
            )

    return selected_windows


def slice_waveforms(filename: str) -> list:
    path = f"{cfg.root_folder}/train_audio/{filename}"
    melspec, waveform = create_log_melspec(path=path)

    if len(waveform) < cfg.sample_rate * 5:
        train_waves = [waveform]
    else:
        selected_windows = detect_train_activity_windows(melspec=melspec)
        train_waves = []
        for window in selected_windows:
            wave_start = int(window[0] / cfg.frame_rate * cfg.sample_rate)
            wave_end = int(window[1] / cfg.frame_rate * cfg.sample_rate)

            if wave_start < 0:
                wave_end = wave_end + abs(wave_start)
                wave_start = 0
            if wave_end > len(waveform):
                wave_end = len(waveform)

            train_waves.append(waveform[wave_start:wave_end])

    return train_waves


def generate_train_waves_and_labels(
    model_input_df: pd.DataFrame,
    class_to_label_map: dict,
) -> list:
    train_waves = [
        slice_waveforms(filename=filename)
        for filename in tqdm(
            model_input_df["filename"],
            desc="Detecting activity and generating train waves",
        )
    ]

    labels = [
        class_to_label_map.get(primary_label)
        for primary_label in tqdm(
            model_input_df["primary_label"], desc="Generating train wave labels"
        )
    ]

    return train_waves, labels


def explode_and_flatten_data(
    model_input_df: pd.DataFrame,
    train_waves: list,
    labels: list,
):
    logging.info("Flattening datasets")
    n_waves_per_waveform = [len(wave) for wave in train_waves]

    model_input_df["n_frames"] = n_waves_per_waveform
    model_input_df = model_input_df.loc[
        model_input_df.index.repeat(n_waves_per_waveform)
    ]

    label_list = np.repeat(labels, n_waves_per_waveform).astype(np.int16)
    train_waves = list(itertools.chain.from_iterable(train_waves))

    return model_input_df, train_waves, label_list


def add_taxonomies(
    model_input_df: pd.DataFrame, taxonomy_df: pd.DataFrame
) -> pd.DataFrame:
    logging.info("Adding bird taxonomy data")
    taxonomy_df["species_code"] = taxonomy_df["SPECIES_CODE"]
    taxonomy_df["species_order"] = taxonomy_df["ORDER1"]
    taxonomy_df["species_family"] = taxonomy_df["FAMILY"]

    model_input_df = model_input_df.merge(
        taxonomy_df[["species_code", "species_order", "species_family"]],
        left_on="primary_label",
        right_on="species_code",
        how="left",
    )

    return model_input_df


def add_durations(model_input_df: pd.DataFrame) -> pd.DataFrame:
    def _get_duration(filename: str) -> float:
        sample = soundfile.SoundFile(
            f"{cfg.data_path}/train_windows_n5_s5_c9/{filename}"
        )
        duration = len(sample) / sample.samplerate
        return duration

    model_input_df["duration"] = [
        _get_duration(filename)
        for filename in tqdm(
            model_input_df["window_filename"], desc="Calculating durations"
        )
    ]

    return model_input_df


def save_data(path: str = cfg.root_folder) -> None:
    logging.info("Saving processed data to disk")
    with open(f"{path}/model_input/train_waves_1.pkl", "wb") as f:
        pickle.dump(train_waves, f)

    with open(f"{path}/model_input/labels_1.pkl", "wb") as f:
        pickle.dump(labels, f)

    model_input_df.to_csv(f"{path}/model_input/model_input_df_1.csv", index=False)


model_input_df, submission_df, taxonomy_df = load_data()
model_input_df = remove_samples(df=model_input_df, filenames_to_drop=filenames_to_drop)

model_input_df, model_input_df_2 = train_test_split(
    model_input_df, shuffle=False, test_size=0.5, random_state=7
)  # in case of OOM issues run processing in 2 parts

visualize_train_activity_detection(model_input_df=model_input_df)

class_to_label_map, label_to_class_map = create_label_mappings(
    submission_df=submission_df
)
train_waves, labels = generate_train_waves_and_labels(
    model_input_df=model_input_df, class_to_label_map=class_to_label_map
)
model_input_df, train_waves, labels = explode_and_flatten_data(
    model_input_df=model_input_df, train_waves=train_waves, labels=labels
)

model_input_df = add_taxonomies(model_input_df=model_input_df, taxonomy_df=taxonomy_df)
model_input_df = add_durations(model_input_df)

save_data()

# #####################################################################################


# due to of OOM issues combine all 2 data parts back into 1
def load_and_combine_processed_data():
    logger.info(f"Loading prepared data parts from {cfg.root_folder}")
    with open(f"{cfg.root_folder}/model_input/train_waves_1.pkl", "rb") as file:
        train_waves_1 = pickle.load(file)
    with open(f"{cfg.root_folder}/model_input/train_waves_2.pkl", "rb") as file:
        train_waves_2 = pickle.load(file)

    with open(f"{cfg.root_folder}/model_input/labels_1.pkl", "rb") as file:
        labels_1 = pickle.load(file)
    with open(f"{cfg.root_folder}/model_input/labels_2.pkl", "rb") as file:
        labels_2 = pickle.load(file)

    model_input_df_1 = pd.read_csv(
        f"{cfg.root_folder}/model_input/model_input_df_1.csv"
    )
    model_input_df_2 = pd.read_csv(
        f"{cfg.root_folder}/model_input/model_input_df_2.csv"
    )
    logger.info(f"Wave size 1: {len(train_waves_1)}, size 2: {len(train_waves_2)}")
    logger.info(f"Label size 1: {len(labels_1)}, size 2: {len(labels_2)}")
    logger.info(f"Df size 1: {len(model_input_df_1)}, size 2: {len(model_input_df_2)}")

    logger.info("Combining data parts")
    train_waves = train_waves_1 + train_waves_2
    del train_waves_1, train_waves_2
    gc.collect()

    labels = np.append(labels_1, labels_2)
    model_input_df = pd.concat([model_input_df_1, model_input_df_2], axis=0)
    model_input_df = model_input_df.reset_index(drop=True)
    del model_input_df_1, model_input_df_2
    del labels_1, labels_2
    gc.collect()

    return train_waves, labels, model_input_df


def save_combined_data(
    path: str,
    train_waves: list,
    labels: list,
    model_input_df: pd.DataFrame,
) -> None:
    logging.info("Saving combined processed data to disk")
    with open(f"{path}/model_input/train_waves.pkl", "wb") as f:
        pickle.dump(train_waves, f)

    with open(f"{path}/model_input/labels.pkl", "wb") as f:
        pickle.dump(labels, f)

    model_input_df.to_csv(f"{path}/model_input/model_input_df.csv", index=False)


train_waves, labels, model_input_df = load_and_combine_processed_data()
# save_combined_data(
#     path=cfg.root_folder,
#     train_waves=train_waves,
#     labels=labels,
#     model_input_df=model_input_df,
# )


######################

for idx, (wave, filename) in tqdm(
    enumerate(zip(train_waves_2, model_input_df_2["filename"])),
    total=len(model_input_df_2),
):
    window_filename = filename.split(".")[0] + f"_{idx}.ogg"
    window_filename = window_filename.replace("/", "_")
    torchaudio.save(
        f"input/birdclef-2024/model_input/train_audio_window_n5_s5_c9/2/{window_filename}",
        torch.tensor(wave).view(1, -1),
        sample_rate=cfg.sample_rate,
        backend="sox",
        format="ogg",
        compression=9,
    )


window_filename = []
for idx, (wave, filename) in enumerate(
    zip(train_waves_1, model_input_df_1["filename"])
):
    window_filename.append(
        "1/" + filename.split(".")[0].replace("/", "_") + f"_{idx}.ogg"
    )
model_input_df_1["window_filename"] = window_filename

window_filename = []
for idx, (wave, filename) in enumerate(
    zip(train_waves_2, model_input_df_2["filename"])
):
    window_filename.append(
        "2/" + filename.split(".")[0].replace("/", "_") + f"_{idx}.ogg"
    )
model_input_df_2["window_filename"] = window_filename

model_input_df = pd.concat([model_input_df_1, model_input_df_2], axis=0)

model_input_df = model_input_df[model_input_df["duration"] >= 2]
model_input_df = model_input_df.reset_index(drop=True)
model_input_df.to_csv("../data/model_input_df.csv", index=False)

# soundfile.write(
#     "test.flac",
#     train_waves[35001],
#     samplerate=cfg.sample_rate,
#     format="flac",
# )
waveform, _ = librosa.load("test_ogg_c9.ogg", sr=cfg.sample_rate)

sns.lineplot(train_waves[35002], linewidth=0.5)
sns.lineplot(waveform, linewidth=0.5)
sns.lineplot(train_waves[35002], linewidth=0.5)

waveform = librosa.util.normalize(waveform)

# import sys
# sys.getsizeof(waveform) / 1024

# # def pcen_bird(melspec):
# #     """
# #     parameters are taken from [1]:
# #         - [1] Lostanlen, et. al. Per-Channel Energy Normalization: Why and How. IEEE Signal Processing Letters, 26(1), 39-43.
# #     """
# #     pcen = librosa.pcen(
# #         melspec * (2**31),
# #         time_constant=0.06,
# #         eps=1e-6,
# #         gain=0.8,
# #         power=0.25,
# #         bias=10,
# #         sr=cfg.sample_rate,
# #         hop_length=cfg.hop_length,
# #     )
# #     return pcen


# # idx = 23388
# # idx = 22848
# # idx = 9599
# # idx = 11818
# # idx = 13755
# # idx = 4612
# # idx = 605
# # idx = 11494


# # visualize_train_activity_detection(model_input_df=model_input_df, idx=23388)


# # select window slices on waveform
# # apply augmentations
# # turn into melspec / log melspec -> use torch audio
# # apply PCEN yes/no
