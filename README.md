# TinyML Visual Navigation

This project recognises printed visual landmarks for GPS-free navigation on a Raspberry Pi 5. The Pi camera feeds a MobileNetV2 classifier; overlapping tiles help detect distant landmarks, CAM estimates their position, and confidence, margin and temporal-vote rules decide whether to issue a steering command. Results and camera frames are published to the project dashboard over MQTT.

The final selected model is the 128 px int8 model. In the day-D evaluation, captured in a new room, it recognised 87% of images with no wrong-place errors and ran at 28 FPS on the Pi CPU. Day D informed several design decisions, and its per-class results are indicative because there are 15 images per class.

## Requirements

Deployment expects a Raspberry Pi 5, Camera Module 3, Picamera2, Python 3, and the TFLite/LiteRT runtime. The `infer.py` fallback supports `tflite_runtime` if `ai-edge-litert` is unavailable. `capture.py` additionally uses OpenCV; the dashboard client uses Paho MQTT. Training notebooks require the TensorFlow/Colab environment used for model training.

On Raspberry Pi OS, install Picamera2 and NumPy through the system package manager. The project was run from a virtual environment configured to see system packages, which avoids breaking Picamera2's binary dependencies. From the project root, install the Python dependencies listed in `requirements.txt`:

```sh
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

If using `tflite_runtime` instead of `ai-edge-litert`, install that runtime for the Pi's Python version and architecture. Set the MQTT password in the environment before starting capture preview or live inference:

```sh
export AETHER_MQTT_PASSWORD="your-rotated-password"
```

Do not commit the password or store it in a tracked `.env` file. `aether_pi.py` reads it when the dashboard client connects and reports a clear error if it is missing.

## Capture data

Run capture from the project root. A session records one class at one height and lighting condition under `data/day_<X>/<class>/`, and appends image metadata to that day's `log.csv`:

```sh
python capture.py --day C --cls pad_A --height 1.0 --lighting window
python capture.py --day C --cls no_target --height 1.5 --lighting lamp --note "wood floor" --repeat
```

Supported classes are `pad_A` through `pad_E` and `no_target`; lighting choices are `window`, `ceiling`, `lamp`, and `evening`. Use `--no-preview` to capture without streaming to the dashboard. Day D is the held-out evaluation set and must not be used for training.

## Run inference

Run from the project root and provide a model file. The JSON config and CAM weights are loaded from the same model directory:

```sh
python infer.py model/128/original/model_128_int8.tflite
```

The runtime currently uses full-frame and 3x3 half-frame tile inference, a 0.85 full-frame threshold, a 0.95 tile threshold, a 0.5 confidence margin, and a 4-of-5 temporal vote. The hand-tuned colour check is disabled for the final system.

## Evaluate and benchmark

Replay the complete decision pipeline on a class-folder dataset such as `data/day_D`:

```sh
python eval_offline.py --directory data/day_D \
  --model model/128/original/model_128_int8.tflite \
  --output offline_scores.csv
```

Benchmark the four original float32/int8 models on the Pi. Add `--camera` to time camera capture as well:

```sh
python benchmark.py \
  --models model/128/original/model_128_f32.tflite \
          model/128/original/model_128_int8.tflite \
          model/96/original/model_96_f32.tflite \
          model/96/original/model_96_int8.tflite \
  --images data/day_D --output benchmark.csv --camera
```

## Training notebooks

Both notebooks are designed for Google Colab. They mount Drive and expect the dataset archive at `MyDrive/tinyml/data.zip`; select a T4 GPU runtime for training.

- [`train_with_validation_and_CV.ipynb`](notebooks/train_with_validation_and_CV.ipynb) runs 5-fold session-grouped cross-validation on the training days, with day D and day E excluded. It saves fold assignments, out-of-fold predictions, metrics, and diagnostic plots under the Drive validation output folder. It uses `FREEZE_BN = True`, so its validation models are not identical to the final deployment model.
- [`full_train.ipynb`](notebooks/full_train.ipynb) trains the deployment model on all eligible sessions, then exports float32 and int8 TensorFlow Lite models, the model configuration, CAM weights, and run metadata to a new dated Drive folder. Its training-set sample is only a sanity check, not an independent accuracy evaluation.

## Project files

`capture.py` records labelled sessions; `config.py` shares camera settings with capture; `infer.py` runs the live camera/model/dashboard pipeline; `aether_pi.py` provides the MQTT client; `eval_offline.py` replays decisions on saved images; `benchmark.py` measures Pi performance; and `plot_day_d.py` generates a day-D plot from an offline score CSV. Model artifacts are in `model/`; the report and figures are in `report.md` and `report_figures/`.
