# Julius Vision

The vision module for [Julius](https://github.com/CHZN1/julius-assistant), a voice assistant for macOS. Today it ships facial-expression recognition: CNNs trained on FER2013 and FER+ (named here only as training data), with the data audit, models, evaluation and figures. It is also the home for Julius's future senses: vision on demand, gestures, presence detection, fall detection and camera-based heart rate. None of those exist yet.

## What is here

| Path | Contents |
|---|---|
| `EMOTE_DATA_COLLECTION_STEP_1.ipynb` | data audit and cleaning |
| `EMOTE_DATA_COLLECTION_ANALYSIS.ipynb` | models, evaluation, error analysis (outputs kept) |
| `figures/` | 14 figures: class balance, mean faces, duplicates, training curves, confusion matrices, per-class F1, label agreement, annotator confidence |
| `artifacts/` | per-run results (`result_*.json`), training histories, cleaning ledger, class maps. Large weights and the FER+ label file are not in git (see below) |
| `emotion/` | experimental live-camera scripts (`webcam_live.py`, `julius_perception.py`) and profiling helpers |

## Results

**Cleaning.** 35,887 images became 33,976 (5.33% removed): blank images, contradictory duplicate labels, train images that also appear in a test split, and validation/test overlap. The full ledger with a reason per step is `artifacts/cleaning_ledger.csv`.

**CNNs from scratch** (about 1.24 M parameters, 7 classes, cleaned FER2013 test split; the majority-class floor is 24.6%):

| Model | Test accuracy | Macro F1 |
|---|---|---|
| A baseline | 55.2% | 0.454 |
| B regularised | 61.5% | 0.537 |
| C deeper | 63.8% | 0.579 |
| D deeper, square-root class weights | 63.6% | **0.595** |

Class weighting trades about 0.2 points of accuracy for a much better rare class: disgust F1 goes from 0.32 (C) to 0.47 (D).

**Transfer learning.** Fine-tuning at 96 px on FER+ soft labels reaches 82.8% accuracy and 0.731 macro F1 on its 3,340-image test split (hard labels: 81.3%, 0.704). That is a different test set and label set from the table above, so the two are not directly comparable. Other runs (RAF-DB, ensembles, a video dataset) are in `artifacts/result_*.json`; read each file's `n_test` before comparing.

## Not included

- Model weights (`*.keras`, about 15 MB each) are too large for git history; they are meant to be attached to a GitHub release (not uploaded yet).
- `artifacts/fer2013new.csv` (FER+ labels) and the FER2013, RAF-DB, CREMA-D and RAVDESS data are third-party; download them from their sources.
- `artifacts/*manifest*.csv` list file names from the audio and face datasets with local paths replaced by `/path/to`.
- `models/face_detection_yunet_2023mar.onnx` is the OpenCV Zoo YuNet face detector (MIT licence), used by `emotion/webcam_live.py`.

## Using it with Julius (experimental)

Julius ships with vision off and a hook: put this repository on `PYTHONPATH` and set `JULIUS_VISION_MODULE=emotion.julius_perception`, a module exposing `Perception` and `describe`. `emotion/julius_perception.py` is that module (camera thread, face detection, emotion smoothing and "check in" gating). It needs the trained weights, `facenet-pytorch`, `insightface`, `opencv-python`, `torch`, `torchvision`, and an enrolled identity embedding that you create yourself. It has not been re-tested since the split.

## License

MIT, see [LICENSE](LICENSE).
