# Window-Non-Window-Glass-Classification-Deep-Neural-Network

# Professional, portfolio-ready project for classifying glass samples into window vs non-window using a Keras-style Deep Neural Network.
This repository provides a clean, reproducible pipeline: data ingestion → preprocessing → model training → evaluation → inference.Window / Non-Window Glass Classification — Deep Neural Network

## Project overview

This project demonstrates an end-to-end supervised classification workflow built for real experimental datasets:

Robust data ingestion (handles common formatting issues and transposed CSVs)

Automated cleaning and type coercion for numeric features

Feature scaling and train/validation/test splitting with reproducibility controls

Keras Sequential DNN (with a scikit-learn fallback option) for reliable training anywhere (local/Colab)

Evaluation artifacts: confusion matrix, classification report, ROC curve, and saved model for inference

The code is written for clarity and reuse so it’s suitable for a GitHub portfolio or production prototyping.

## Key features

Single-script run: model.py performs preprocessing, training, evaluation and saves artifacts.

predict.py for fast inference on new CSV files.

keras_colab.py: plug-and-play Colab-ready script (installs TF and trains on Colab/GPU).

Auto-detection for transposed inputs (the pipeline will transpose if data are stored in the transposed format found in many lab exports).

Outputs saved to outputs/; trained model and runtime artifacts saved to saved_model/.

Reproducible splits with --seed argument.

## Data expectations

The pipeline accepts two CSV files:

inputs_glass.csv — feature table (each row = a sample; columns = features such as Refractive_Index, Sodium, Magnesium, Aluminium, Silicon, Potassium, Calcium, Barium, Iron, etc.).

target_glass.csv — corresponding labels (0/1 or non-window/window encoded numerically).

Robustness features:

If your CSVs are in the transposed format (rows/columns swapped, as commonly exported from some lab systems), the pipeline auto-detects and transposes them before processing.

Non-numeric characters, unit markers, stray asterisks, and commas are cleaned automatically before coercion to numeric types.

## Model & training (brief)

Architecture (default): Normalization -> Dense(32) -> Dense(64) -> Dense(128) -> Dense(256) -> Dense(128) -> Dense(64) -> Dense(32) -> Dense(1, sigmoid)

Loss: binary_crossentropy

Optimizer: rmsprop or adam (selectable)

Metrics: accuracy (plus precision/recall/F1 via scikit-learn for reporting)

Early stopping and best-model checkpointing included (when running in Keras mode)

## Note: The repository supports two execution modes:

Keras/TensorFlow (preferred; GPU accelerate on Colab)

scikit-learn MLPClassifier fallback (for environments without TF)

## Quickstart — local

Clone the repository or create a folder and copy files.

Put your inputs_glass.csv and target_glass.csv under data/.

Create and activate a Python virtual environment:

python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt


## Train the model:

python model.py --inputs data/inputs_glass.csv --targets data/target_glass.csv --epochs 100 --batch 16 --seed 42


## After completion:

outputs/classification_report.txt — classification metrics

outputs/confusion_matrix.png — confusion matrix visualization

outputs/roc_curve.png — ROC curve and AUC

saved_model/ — trained model file (.h5 if TF used or .joblib if sklearn fallback), scaler and feature list

## Quickstart — Google Colab

Open keras_colab.py in a Colab cell or simply upload your CSVs to Colab and run:

# in Colab (cell)
!python keras_colab.py


The Colab script installs TensorFlow and trains with GPU support. It also saves welding_dnn_colab.h5 (or similarly named file) to the Colab filesystem — remember to copy it to Drive if you want persistence.

## Inference

Once you have saved_model/feature_columns.joblib and the saved model + scaler:

python predict.py --model saved_model/glass_dnn.h5 --scaler saved_model/scaler.joblib --input new_samples.csv


The script produces outputs/predictions.csv containing predicted probabilities and labels for each numeric sample row in new_samples.csv.

## Reproducibility & tuning tips

Use --seed to fix seed for train/test splits.

Change --epochs, --batch from the CLI to tune training runs.

For imbalanced classes, enable class weights or resampling (code comments show where to add this).

For production deployment, export the model + scaler and use predict.py or convert the model to TensorFlow SavedModel.

## Dependencies

Minimal requirements:

pandas
numpy
matplotlib
seaborn
scikit-learn
tensorflow>=2.9   # optional; fallback uses scikit-learn if TF absent
joblib


Install either full TF or rely on the sklearn fallback for quick local runs.
