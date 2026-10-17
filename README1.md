# AI in Security: Network Intrusion Detection

A machine learning project implementing Random Forest classifiers to detect and categorize network traffic attacks. This project trains both a binary classifier (Attack vs. Normal) and a multiclass classifier (9 specific attack categories + Normal) using the UNSW-NB15 network traffic dataset.

## Dataset

* **Source:** `UNSW_NB15_balanced_30k.csv`
* **Features:** 42 network traffic features (39 numeric, 3 categorical: `proto`, `service`, `state`).
* **Targets:**
* `label` (Binary: 0 for Normal, 1 for Attack)
* `attack_cat` (Multiclass: Normal, Analysis, Backdoor, DoS, Exploits, Fuzzers, Generic, Reconnaissance, Shellcode, Worms).



## Dependencies

Ensure you have Python 3.x installed along with the following libraries:

* `scikit-learn`
* `pandas`
* `numpy`
* `joblib`

Install dependencies via pip:

```bash
pip install scikit-learn pandas numpy joblib

```

## Project Structure

* `sk11321_rf.py`: The main Python script containing the data preprocessing, model training, evaluation, and export logic.
* `UNSW_NB15_balanced_30k.csv`: The training and evaluation dataset (must be present in the same directory).
* `randomForestModel.joblib`: The generated model artifact containing the fully trained binary and multiclass pipelines.

## Model Architecture

The project utilizes `scikit-learn` Pipelines to combine preprocessing and classification into a single exportable artifact.

1. **Preprocessor:** A `ColumnTransformer` applies `OneHotEncoder` to the categorical features while passing numeric features through unchanged.
2. **Binary Pipeline:** A lightweight `RandomForestClassifier` optimized for high accuracy and F1 score on the binary `label`.
3. **Multiclass Pipeline:** A heavily tuned `RandomForestClassifier` utilizing class weighting (`class_weight='balanced'`), strict node pruning, and feature subsampling to handle extreme class imbalances (e.g., catching rare `Worms` and `Shellcode` attacks) while preventing false-positive hijacking.

## Usage

To train the models, evaluate their performance on a local 80/20 split, and generate the final `.joblib` artifact (refitted on the entire 30,000-row dataset):

```bash
python3 sk11321_rf.py

```

The script will output local Precision, Recall, and F1 scores to the console and save the bundled models to `randomForestModel.joblib`. The final artifact is optimized to remain under a strict 12 MB file size limit for production deployment.
