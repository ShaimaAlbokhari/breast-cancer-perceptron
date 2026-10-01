# 🧠 Breast Cancer Classification with Perceptron

A binary breast cancer classification project using a **Perceptron implemented from scratch with NumPy** and evaluated on the **Wisconsin Diagnostic Breast Cancer (WDBC) dataset**.

The model classifies breast tumors as **Benign (B)** or **Malignant (M)** using 30 numerical features extracted from digitized images of fine needle aspirates of breast masses.

---

## 🎯 Project Objective

The objective of this project is to implement and evaluate a binary Perceptron classifier without relying on a pre-built machine learning classification model.

The Perceptron learning algorithm, prediction function, weight updates, bias updates, and accuracy calculations are implemented directly using NumPy.

`scikit-learn` is used only for splitting the dataset into training and testing sets.

---

## 📊 Dataset

The project uses the **Breast Cancer Wisconsin (Diagnostic) dataset**.

The dataset contains:

- 569 samples
- 30 numerical features
- Two diagnosis classes:
  - **B — Benign**
  - **M — Malignant**

The original dataset format contains:

```text
Column 0  → ID
Column 1  → Diagnosis
Columns 2+ → Numerical features
```

---

## ⚙️ Data Preprocessing

Before training the model:

1. The ID column is excluded.
2. Diagnosis labels are converted into binary values:
   - Benign → `0`
   - Malignant → `1`
3. The dataset is split into:
   - 80% training data
   - 20% testing data
4. Stratified splitting is used to preserve the class distribution.
5. Features are standardized using the mean and standard deviation calculated from the **training data only**.

Using training statistics for normalization prevents information from the test set from leaking into the training process.

---

## 🧠 Perceptron Implementation

The binary Perceptron is implemented from scratch using NumPy.

The model calculates:

```text
z = X · W + b
```

The predicted class is:

```text
1 if z >= 0
0 otherwise
```

During training, the weights and bias are updated according to the prediction error.

### Model Configuration

```text
Learning Rate: 0.01
Epochs: 100
Train/Test Split: 80/20
Random State: 42
```

Training samples are shuffled during every epoch using a reproducible random number generator.

---

## 📈 Results

After 100 training epochs, the final results were:

| Metric | Result |
|---|---:|
| Training Accuracy | **98.68%** |
| Test Accuracy | **96.49%** |

### Confusion Matrix

```text
                 Predicted
                 B     M
Actual B        71     1
Actual M         3    39
```

The confusion matrix corresponds to:

- True Negatives: **71**
- False Positives: **1**
- False Negatives: **3**
- True Positives: **39**

---

## 📉 Visualizations

The project generates two visualizations:

### Accuracy Across Epochs

Training and testing accuracy are recorded after every epoch and plotted to show the model's performance throughout training.

### Confusion Matrix

A confusion matrix is visualized to show the number of correctly and incorrectly classified benign and malignant samples.

---

## 🛠️ Technologies

- Python
- NumPy
- Matplotlib
- scikit-learn

---

## 📁 Project Structure

```text
breast-cancer-perceptron/
│
├── breast_cancer_perceptron.py
├── breast_cancer_perceptron.ipynb
├── wdbc.data
└── README.md
```

---

## ▶️ Running the Project

Install the required libraries:

```bash
pip install numpy matplotlib scikit-learn
```

Then run:

```bash
python breast_cancer_perceptron.py
```

The program will:

1. Load and preprocess the dataset.
2. Train the Perceptron for 100 epochs.
3. Display training and testing accuracy.
4. Calculate the final confusion matrix.
5. Plot the accuracy history.
6. Visualize the confusion matrix.

---

## 📚 Dataset Attribution

This project uses the **Breast Cancer Wisconsin (Diagnostic)** dataset from the UCI Machine Learning Repository.

Dataset created by:

- Wolberg, William
- Mangasarian, Olvi
- Street, Nick
- Street, W.

Source: UCI Machine Learning Repository  
Dataset: Breast Cancer Wisconsin (Diagnostic)

The dataset is used in accordance with its associated license and attribution requirements.

---

## 👩‍💻 Developer

**Shaima Albokhari**  
B.Sc. Computer Science — Taif University
