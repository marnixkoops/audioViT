# def create_waveform_augmentations(
#     augment_waveforms: bool = cfg.augment_waveforms,
# ) -> A.Compose:
#     if augment_waveforms:
#         waveform_augmentations = A.Compose(
#             [
#                 A.Compose(
#                     [
#                         A.Gain(min_gain_db=-10, max_gain_db=10, p=0.5),
#                         A.LowPassFilter(
#                             min_cutoff_freq=20,
#                             max_cutoff_freq=5000,
#                             min_rolloff=12,
#                             max_rolloff=24,
#                             zero_phase=False,
#                             p=0.5,
#                         ),
#                         A.AddGaussianNoise(
#                             min_amplitude=0.001, max_amplitude=0.03, p=0.2
#                         ),
#                         A.GainTransition(
#                             min_gain_db=-10,
#                             max_gain_db=10,
#                             min_duration=0.2,
#                             max_duration=3,
#                             p=0.2,
#                         ),
#                         # A.TimeMask(
#                         #     min_band_part=0.1, max_band_part=0.25, fade=True, p=0.5
#                         # ),
#                         # A.Reverse(p=0.2),
#                     ],
#                 ),
#                 # A.SomeOf(
#                 #     (1, 3),
#                 #     [
#                 #         A.SevenBandParametricEQ(min_gain_db=-10, max_gain_db=10, p=0.2),
#                 #         A.BandPassFilter(
#                 #             min_center_freq=100, max_center_freq=6000, p=0.2
#                 #         ),
#                 #         A.AirAbsorption(
#                 #             min_distance=10,
#                 #             max_distance=100,
#                 #             p=0.2,
#                 #         ),
#                 #     ],
#                 # ),
#                 # A.SomeOf(
#                 #     (0, 1),
#                 #     [
#                 #         # A.BitCrush(min_bit_depth=5, max_bit_depth=14, p=0.2),
#                 #         A.PitchShift(min_semitones=-5.0, max_semitones=5.0, p=0.2),
#                 # #         A.TimeMask(
#                 # #             min_band_part=0.1, max_band_part=0.15, fade=True, p=0.2
#                 # #         ),
#                 #     ],
#                 # ),
#             ]
#         )
#     else:
#         waveform_augmentations = None

#     # logger.info("Augmentations:")
#     # for transform in train_augmentations.transforms:
#     #     logger.info(f"{transform.__class__.__name__}")

#     return waveform_augmentations


# def create_melspec_augmentations(augment_melspecs: bool = cfg.augment_melspecs):
#     if augment_melspecs:
#         melspec_augmentations = albumentations.Compose(
#             [
#                 albumentations.GaussNoise(var_limit=5 / 255, p=0.2),
#                 albumentations.ImageCompression(
#                     quality_lower=80, quality_upper=100, p=0.2
#                 ),
#                 albumentations.CoarseDropout(
#                     max_holes=4, max_height=20, max_width=20, p=0.2
#                 ),
#                 albumentations.XYMasking(
#                     p=0.3,
#                     num_masks_x=(1, 3),
#                     num_masks_y=(1, 3),
#                     mask_x_length=(2, 20),
#                     mask_y_length=(2, 10),
#                 ),
#             ]
#         )
#     else:
#         melspec_augmentations = None

#     return melspec_augmentations


melspec_augmentations = A.Compose(
    [
        A.HorizontalFlip(p=0.3),
        A.CoarseDropout(
            max_height=50,
            max_width=50,
            max_holes=1,
            p=0.5,
        ),
        A.GaussNoise(var_limit=5 / 255, p=0.2),
        A.ImageCompression(quality_lower=80, quality_upper=100, p=0.2),
        A.XYMasking(
            p=0.3,
            num_masks_x=(1, 3),
            num_masks_y=(1, 3),
            mask_x_length=(2, 20),
            mask_y_length=(2, 10),
        ),
    ]
)


def create_melspec_augmentations(augment_melspecs: bool = cfg.augment_melspecs):
    if augment_melspecs:
        melspec_augmentations = albumentations.Compose(
            [
                albumentations.GaussNoise(var_limit=5 / 255, p=0.2),
                albumentations.ImageCompression(
                    quality_lower=80, quality_upper=100, p=0.2
                ),
                albumentations.CoarseDropout(
                    max_holes=4, max_height=20, max_width=20, p=0.2
                ),
                albumentations.XYMasking(
                    p=0.3,
                    num_masks_x=(1, 3),
                    num_masks_y=(1, 3),
                    mask_x_length=(2, 20),
                    mask_y_length=(2, 10),
                ),
            ]
        )
    else:
        melspec_augmentations = None

    return melspec_augmentations
