# Gender Classification with CNN (PyTorch)

A convolutional neural network built with PyTorch to classify face images as **male** or **female**. This project was built as a learning exercise in CNN architecture design, training, and — most importantly — diagnosing and fixing overfitting and class bias.

## Overview

- **Framework:** PyTorch + torchvision
- **Task:** Binary image classification (male / female)
- **Input size:** 224x224 RGB images
- **Dataset structure:** `ImageFolder` format (`data/Training`, `data/Validation`)

## Model Architecture

A custom CNN (`Gender`) with:
- 3 convolutional blocks (Conv2d → ReLU → MaxPool2d), channels: 3 → 32 → 64 → 128
- `AdaptiveAvgPool2d` to reduce spatial dimensions before the fully connected layers (keeps parameter count low and reduces overfitting compared to a full flatten)
- Fully connected layers with dropout for regularization
- Output: 2 classes (female, male)

## Training Setup

- **Loss:** `CrossEntropyLoss`
- **Optimizer:** Adam (`lr=0.001`)
- **Batch size:** 32
- **Epochs:** 15
- **Data augmentation (training only):** random horizontal flip, small random rotation

## Results

Final model performance on the validation set:

| Metric | Value |
|---|---|
| Validation Accuracy | 91.87% |
| Validation Loss | 0.2012 |

| Class | Precision | Recall | F1-score |
|---|---|---|---|
| Female | 0.93 | 0.90 | 0.92 |
| Male | 0.90 | 0.94 | 0.92 |

## Key Lessons Learned

This project intentionally went through a full debugging cycle rather than stopping at the first "good" accuracy number:

1. **Initial model overfit heavily** — 95.76% val accuracy but a large train/val loss gap (0.017 vs 0.228), meaning the model was overconfident and poorly calibrated despite being accurate.
2. **First regularization pass overcorrected** — adding dropout, weight decay, label smoothing, and strong augmentation all at once closed the loss gap but dropped accuracy to 87–91% and introduced a **class bias** (model became much better at recognizing "female" than "male").
3. **Iterative tuning** — removing overly strong regularization (`ColorJitter`, weight decay, label smoothing) restored balance across both classes while keeping the improved architecture (smaller FC layer + dropout), landing on a model that is both reasonably accurate (91.87%) and fair across classes.

The confusion matrix and per-class precision/recall were essential here — overall accuracy alone hid the class bias introduced during regularization.

## Real-World Testing Notes

Validation accuracy (91.87%) reflects performance on images drawn from the same distribution as training data. Manually testing the served model on real, unseen photos revealed a pattern worth documenting:

| Photo framing | Result |
|---|---|
| Frontal, face-forward, closer crop | Correct, higher confidence (~70%) |
| Angled/tilted, full torso, arms in frame | Incorrect, lower confidence (~60-65%) |
| Side profile, full body, background clutter | Incorrect, lower confidence (~60-65%) |

**Takeaway:** the model performs best on frontal, face-centered images similar to the training distribution, and accuracy degrades on angled, full-body, or candid shots with background clutter. Confidence tended to track correctness (right answers came with higher confidence than wrong ones), suggesting reasonable — though imperfect — calibration. This is a good example of why validation accuracy alone doesn't capture real-world reliability; a held-out validation set drawn from the same source as training data can't reveal distribution sensitivity that only shows up with genuinely new, differently-framed images.

## Usage

### Training
Run the notebook (`gender_classification.ipynb`) top to bottom. It will:
1. Load training/validation data from `data/Training` and `data/Validation`
2. Train the model for the configured number of epochs
3. Print train/validation loss and accuracy per epoch

### Serving predictions with Flask
The `app/` folder contains a small Flask app that loads the saved weights and serves predictions through a web UI and a JSON API.

```bash
cd app
pip install -r requirements.txt
python app.py
```

Then open `http://localhost:5000` for the upload UI, or send a POST request with an image to the `/predict` endpoint to use the API directly.

`app/model.py` defines the `Gender` class and must exactly match the architecture used during training — this is what lets `model.load_state_dict()` restore the saved weights correctly.

### Inference on a new image (standalone script)
For running inference outside the Flask app, load `Gender` from `model.py`, restore the saved weights, apply the same resize/tensor transform used during training, and run a forward pass with `torch.no_grad()` to get the predicted class and confidence.

## Limitations

- Binary classification only (male/female) — does not represent gender as a spectrum.
- Trained on a specific dataset; performance may not generalize well to different demographics, lighting conditions, or image sources not represented in training data.
- **Framing-sensitive:** performs best on frontal, face-centered photos. Accuracy drops on angled, full-body, or candid images with background clutter (see Real-World Testing Notes above).
- Built for educational purposes — not intended for use in any high-stakes, biometric, surveillance, or identity-verification context.

## Requirements

```
torch
torchvision
flask
pillow
pandas
matplotlib
numpy
scikit-learn
seaborn
```

## Project Structure

```
├── gender_classification.ipynb   # Main notebook: data loading, model, training, evaluation
├── data/
│   ├── Training/
│   └── Validation/
├── app/
│   ├── app.py                    # Flask API + web UI
│   ├── model.py                  # Gender CNN class (must match training architecture)
│   ├── gender_model.pth          # Saved model weights
│   ├── templates/
│   │   └── index.html            # Upload UI
│   └── static/
│       └── style.css
└── README.md
```