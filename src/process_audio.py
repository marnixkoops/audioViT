import gc
import itertools
import logging
import os
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
from joblib import Parallel, delayed
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
    root_folder = "../data"
    normalize_waveform = False
    sample_rate = 32000
    n_fft = 2048
    hop_length = 512
    window_length = None
    melspec_hres = 128
    melspec_wres = 312
    freq_min = 20
    freq_max = 16000
    log_scale_power = 2
    create_frames = True
    frame_duration = 6
    max_decibels = 80
    frame_rate = sample_rate / hop_length
    max_load_duration = 5 * 60
    n_workers = os.cpu_count() - 4


def load_metadata(path: str = cfg.root_folder) -> pd.DataFrame:
    logging.info("Loading data")
    model_input_df = pd.read_csv(
        f"{cfg.root_folder}/train_metadata.csv",
        dtype={
            "secondary_labels": "string",
            "primary_label": "category",
        },
    )

    model_input_additional_df = pd.read_csv(
        f"{cfg.root_folder}/train_metadata_additional.csv",
    )

    missing_files = [
        "XC775312",
        "XC881009",
        "XC891005",
        "XC891004",
        "XC798809",
        "XC798808",
        "XC798807",
        "XC798806",
        "XC798805",
        "XC835367",
        "XC762524",
    ]
    model_input_additional_df = model_input_additional_df[
        ~model_input_additional_df["file"].isin(missing_files)
    ]

    model_input_additional_df["filename"] = (
        model_input_additional_df["primary_label"]
        + "/"
        + model_input_additional_df["file"]
        + ".wav"
    )
    model_input_additional_df = model_input_additional_df[
        ["primary_label", "also", "type", "lat", "lng", "rec", "lic", "url", "filename"]
    ]
    model_input_additional_df.columns = [
        "primary_label",
        "secondary_labels",
        "type",
        "latitude",
        "longitude",
        "author",
        "license",
        "url",
        "filename",
    ]

    model_input_df = pd.concat([model_input_df, model_input_additional_df], axis=0)
    model_input_df = model_input_df.reset_index(drop=True)

    submission_df = pd.read_csv(f"{cfg.root_folder}/sample_submission.csv")
    taxonomy_df = pd.read_csv(f"{cfg.root_folder}/eBird_Taxonomy_v2021.csv")

    return model_input_df, submission_df, taxonomy_df


def remove_samples(
    df: pd.DataFrame,
    filenames_to_drop: list[str],
    remove_duplicates: bool = False,
) -> pd.DataFrame:
    if remove_duplicates:
        logging.info("Removing hand-selected samples from data")
        df = df[~df["filename"].isin(filenames_to_drop)]
    df = df.reset_index(drop=True)
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
    waveform, _ = librosa.load(path, sr=cfg.sample_rate, duration=cfg.max_load_duration)

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

    top_peaks = np.argsort(-energy[peaks])[:3]
    top_peaks_index = peaks[top_peaks]

    selected_windows = []
    for i in top_peaks_index:
        selected_windows.append([i - (cfg.frame_rate * 3), i + (cfg.frame_rate * 3)])

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


def detect_train_activity_windows(
    melspec: np.ndarray, n_peaks: int = 5, always_include_first_secs: bool = False
) -> list:
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

    if peak_idx.size == 0:  # if no peaks are detected take the first 6 seconds
        selected_windows = [[0, cfg.frame_rate * 6]]
    else:
        selected_windows = []
        if always_include_first_secs:
            selected_windows.append([0, cfg.frame_rate * 6])

        top_peaks = np.argsort(-energy_per_frame[peak_idx])[:n_peaks]
        top_peaks_idx = peak_idx[top_peaks]
        for peak in top_peaks_idx:
            selected_windows.append(
                [peak - (cfg.frame_rate * 3), peak + (cfg.frame_rate * 3)]
            )

    return selected_windows


def slice_waveforms(filename: str) -> list:
    path = f"{cfg.root_folder}/train_audio/{filename}"
    melspec, waveform = create_log_melspec(path=path)
    duration = len(waveform)

    def _select_slices(selected_windows: list) -> list:
        train_waves = []
        for window in selected_windows:
            wave_start = int(window[0] / cfg.frame_rate * cfg.sample_rate)
            wave_end = int(window[1] / cfg.frame_rate * cfg.sample_rate)

            if wave_start < 0:
                wave_end = wave_end + abs(wave_start)
                wave_start = 0
            if wave_end > duration:
                wave_end = duration

            train_waves.append(waveform[wave_start:wave_end])

        return train_waves

    if duration / cfg.sample_rate < 6:
        train_waves = [waveform]
    elif duration / cfg.sample_rate < 12:
        selected_windows = detect_train_activity_windows(melspec=melspec, n_peaks=2)
        train_waves = _select_slices(selected_windows)
    elif duration / cfg.sample_rate > 90:
        selected_windows = detect_train_activity_windows(melspec=melspec, n_peaks=4)
        train_waves = _select_slices(selected_windows)
    elif duration / cfg.sample_rate > 180:
        selected_windows = detect_train_activity_windows(melspec=melspec, n_peaks=5)
        train_waves = _select_slices(selected_windows)
    else:
        selected_windows = detect_train_activity_windows(melspec=melspec, n_peaks=3)
        train_waves = _select_slices(selected_windows)

    return train_waves


def generate_train_waves(
    model_input_df: pd.DataFrame,
) -> list:
    train_waves = Parallel(n_jobs=cfg.n_workers)(
        delayed(slice_waveforms)(filename=filename)
        for filename in tqdm(
            model_input_df["filename"],
            desc="Detecting activity and generating train waves",
        )
    )
    return train_waves


def explode_and_flatten_data(
    model_input_df: pd.DataFrame,
    train_waves: list,
):
    logging.info("Flattening datasets")
    n_waves_per_waveform = [len(wave) for wave in train_waves]

    model_input_df["n_frames"] = n_waves_per_waveform
    model_input_df = model_input_df.reset_index(drop=False, names="sample_index")

    model_input_df = model_input_df.loc[
        model_input_df.index.repeat(n_waves_per_waveform)
    ]
    model_input_df = model_input_df.reset_index(drop=True)

    train_waves = list(itertools.chain.from_iterable(train_waves))

    return model_input_df, train_waves


def pad_or_crop_waveforms(waveforms: list, pad_method: str = "repeat") -> list:
    logging.info("Padding or cropping waveforms to desired duration")
    desired_length = cfg.sample_rate * cfg.frame_duration

    def _pad_or_crop(waveform: np.ndarray) -> np.ndarray:
        length = len(waveform)

        while length < desired_length:  # repeat if waveform too small
            repeat_length = desired_length - length
            padding_array = waveform[:repeat_length]
            if pad_method != "repeat":
                padding_array = np.zeros(shape=waveform[:repeat_length].shape)
            waveform = np.concatenate([waveform, padding_array])
            length = len(waveform)

        if length > desired_length:  # crop if waveform is too big
            offset = np.random.randint(0, length - desired_length)
            waveform = waveform[offset : offset + desired_length]

        return waveform

    waveforms = [_pad_or_crop(wave) for wave in tqdm(waveforms, desc="Padding waves")]

    return waveforms


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
        sample = soundfile.SoundFile(f"{cfg.root_folder}/train_audio/{filename}")
        duration = len(sample) / sample.samplerate
        return duration

    model_input_df["duration"] = [
        _get_duration(filename)
        for filename in tqdm(model_input_df["filename"], desc="Calculating durations")
    ]

    return model_input_df


def save_data(path: str = cfg.root_folder) -> None:
    logging.info("Saving processed data to disk")
    with open(f"{path}/train_waves_1.pkl", "wb") as f:
        pickle.dump(train_waves_1, f)
    model_input_df_2.to_csv(f"{path}/model_input_df_2.csv", index=False)


if __name__ == "__main__":
    model_input_df, submission_df, taxonomy_df = load_metadata()
    model_input_df = remove_samples(
        df=model_input_df, filenames_to_drop=filenames_to_drop
    )
    model_input_df = add_taxonomies(
        model_input_df=model_input_df, taxonomy_df=taxonomy_df
    )

    # not enough local RAM, to prevent OOM issues run processing in 3 parts
    model_input_df, model_input_df_1 = train_test_split(
        model_input_df, shuffle=False, test_size=0.33, random_state=7
    )
    model_input_df_2, model_input_df_3 = train_test_split(
        model_input_df, shuffle=False, test_size=0.33, random_state=7
    )

    visualize_train_activity_detection(model_input_df=model_input_df)

    # part 1
    train_waves_1 = generate_train_waves(model_input_df=model_input_df_1)
    model_input_df_1, train_waves_1 = explode_and_flatten_data(
        model_input_df=model_input_df_1, train_waves=train_waves_1
    )
    logging.info("Saving processed data to disk")

    window_filename = []
    for idx, (wave, filename) in enumerate(
        zip(train_waves_1, model_input_df_1["filename"])
    ):
        window_filename.append(
            "1/" + filename.split(".")[0].replace("/", "_") + f"_{idx}.ogg"
        )
    model_input_df_1["window_filename"] = window_filename
    model_input_df_1.to_csv("../data/model_input_df_1.csv", index=False)

    for idx, (wave, filename) in tqdm(
        enumerate(zip(train_waves_1, model_input_df_1["filename"])),
        total=len(model_input_df_1),
    ):
        window_filename = filename.split(".")[0] + f"_{idx}.ogg"
        window_filename = window_filename.replace("/", "_")
        torchaudio.save(
            f"{cfg.root_folder}/train_windows_n3/1/{window_filename}",
            torch.tensor(wave).view(1, -1),
            sample_rate=cfg.sample_rate,
            backend="sox",
            format="ogg",
            compression=10,
        )

    del train_waves_1
    gc.collect()

    # part 2
    train_waves_2 = generate_train_waves(model_input_df=model_input_df_2)
    model_input_df_2, train_waves_2 = explode_and_flatten_data(
        model_input_df=model_input_df_2, train_waves=train_waves_2
    )

    window_filename = []
    for idx, (wave, filename) in enumerate(
        zip(train_waves_2, model_input_df_2["filename"])
    ):
        window_filename.append(
            "2/" + filename.split(".")[0].replace("/", "_") + f"_{idx}.ogg"
        )
    model_input_df_2["window_filename"] = window_filename
    model_input_df_2.to_csv("../data/model_input_df_2.csv", index=False)

    for idx, (wave, filename) in tqdm(
        enumerate(zip(train_waves_2, model_input_df_2["filename"])),
        total=len(model_input_df_2),
    ):
        window_filename = filename.split(".")[0] + f"_{idx}.ogg"
        window_filename = window_filename.replace("/", "_")
        torchaudio.save(
            f"{cfg.root_folder}/train_windows_n3/2/{window_filename}",
            torch.tensor(wave).view(1, -1),
            sample_rate=cfg.sample_rate,
            backend="sox",
            format="ogg",
            compression=10,
        )

    del train_waves_2
    gc.collect()

    # part 3
    train_waves_3 = generate_train_waves(model_input_df=model_input_df_3)
    model_input_df_3, train_waves_3 = explode_and_flatten_data(
        model_input_df=model_input_df_3, train_waves=train_waves_3
    )

    window_filename = []
    for idx, (wave, filename) in enumerate(
        zip(train_waves_3, model_input_df_3["filename"])
    ):
        window_filename.append(
            "3/" + filename.split(".")[0].replace("/", "_") + f"_{idx}.ogg"
        )
    model_input_df_3["window_filename"] = window_filename
    model_input_df_3.to_csv("../data/model_input_df_3.csv", index=False)

    for idx, (wave, filename) in tqdm(
        enumerate(zip(train_waves_3, model_input_df_3["filename"])),
        total=len(model_input_df_3),
    ):
        window_filename = filename.split(".")[0] + f"_{idx}.ogg"
        window_filename = window_filename.replace("/", "_")
        torchaudio.save(
            f"{cfg.root_folder}/train_windows_n3/3/{window_filename}",
            torch.tensor(wave).view(1, -1),
            sample_rate=cfg.sample_rate,
            backend="sox",
            format="ogg",
            compression=10,
        )

    del train_waves_3
    gc.collect()

    # create full model_input_df
    model_input_df_1 = pd.read_csv(f"{cfg.root_folder}/model_input_df_1.csv")
    model_input_df_2 = pd.read_csv(f"{cfg.root_folder}/model_input_df_2.csv")
    model_input_df_3 = pd.read_csv(f"{cfg.root_folder}/model_input_df_3.csv")
    model_input_df = pd.concat(
        [model_input_df_1, model_input_df_2, model_input_df_3], axis=0
    )
    model_input_df.to_csv(f"{cfg.root_folder}/model_input_df.csv", index=False)
