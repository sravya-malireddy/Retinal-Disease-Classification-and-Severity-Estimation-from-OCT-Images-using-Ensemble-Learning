# Retinal Disease Classification and Severity Estimation from OCT Images using Ensemble Learning

A deep learning-based system for classifying retinal diseases from Optical Coherence Tomography (OCT) images using transfer learning, ensemble learning, Test-Time Augmentation (TTA), and Grad-CAM explainability.

This project was developed as a major academic project in Electronics and Communication Engineering.

---

## Overview

Retinal OCT images contain detailed information about the internal structure of the retina. However, analyzing large numbers of OCT scans manually can be time-consuming and requires specialized expertise.

This project explores an automated deep learning pipeline that processes OCT images and classifies them into four categories:

- CNV — Choroidal Neovascularization
- DME — Diabetic Macular Edema
- DRUSEN
- NORMAL

The system combines predictions from multiple pretrained convolutional neural networks instead of depending on a single model.

The final pipeline includes:

1. Image preprocessing
2. Data augmentation
3. Transfer learning
4. Model training and fine-tuning
5. Test-Time Augmentation
6. Weighted ensemble prediction
7. Performance evaluation
8. Grad-CAM visualization
9. Confidence-based severity estimation

---

## System Architecture

```text
                 OCT Image
                     |
                     v
          +----------------------+
          | Image Preprocessing  |
          | Resize / Normalize   |
          | Augmentation         |
          +----------+-----------+
                     |
                     v
              +-------------+
              |   Dataset   |
              +------+------+
                     |
        +------------+------------+
        |            |            |
        v            v            v
   DenseNet121   EfficientNetB4  ResNet50V2
        |            |            |
        v            v            v
   Probability   Probability   Probability
     Scores         Scores        Scores
        |            |            |
        +------------+------------+
                     |
                     v
          Weighted Ensemble Model
                     |
                     v
             Test-Time Augmentation
                     |
                     v
              Final Prediction
                     |
        +------------+-------------+
        |            |             |
        v            v             v
   Disease Class  Confidence   Severity
        |
        v
     Grad-CAM
        |
        v
   Visual Explanation
```

---

## Models Used

The project uses transfer learning with three pretrained CNN architectures:

| Model | Purpose |
|---|---|
| DenseNet121 | Feature extraction and classification |
| EfficientNetB4 | Efficient feature representation |
| ResNet50V2 | Residual feature learning |
| Weighted Ensemble | Combines predictions from all three models |

The models are initialized using pretrained ImageNet weights and subsequently adapted to the four-class OCT classification problem.

---

## Dataset

The project uses a subset of the publicly available **Retinal OCT (C8) dataset**.

The four classes used in the project are:

```text
CNV
DME
DRUSEN
NORMAL
```

The dataset is divided into training, validation, and testing data.

The training data is further split into training and validation subsets using stratified sampling.

The images are resized to:

```text
256 × 256 pixels
```

---

## Data Preprocessing

The preprocessing pipeline includes:

- Image resizing
- RGB conversion
- Pixel normalization
- Horizontal flipping
- Vertical flipping
- Rotation
- Brightness and contrast adjustment
- Gaussian blur
- CLAHE enhancement

A custom data-processing pipeline is used to prepare the images before model training.

The project uses a fixed random seed to improve reproducibility.

---

## Transfer Learning

The three CNN models are initialized using pretrained ImageNet weights.

The training process follows a transfer-learning approach:

1. Load pretrained CNN architecture
2. Freeze the initial layers
3. Add a task-specific classification layer
4. Train the classification layers
5. Unfreeze selected upper layers
6. Fine-tune the model using OCT images

The models use categorical cross-entropy loss and the Adam optimizer.

---

## Ensemble Learning

Instead of using the prediction of a single CNN, the project combines the probability outputs of all three models.

The final prediction is calculated using a weighted average:

```text
P_final =
0.3 × P_DenseNet
+
0.3 × P_EfficientNet
+
0.4 × P_ResNet
```

The weights used in the implementation are:

| Model | Weight |
|---|---:|
| DenseNet121 | 0.30 |
| EfficientNetB4 | 0.30 |
| ResNet50V2 | 0.40 |

This allows the final model to use information from multiple architectures.

---

## Test-Time Augmentation

Test-Time Augmentation (TTA) is applied during inference.

For each test batch, predictions are generated from:

- Original image
- Horizontally flipped image
- Brightness-adjusted image

The predictions are then averaged before the ensemble stage.

This is intended to make predictions more stable under small variations in the input image.

---

## Explainability with Grad-CAM

Grad-CAM is used to visualize regions of the OCT image that influence the model's prediction.

The resulting heatmaps provide a visual representation of the areas receiving greater attention from the CNN.

This makes the system easier to interpret compared with a classification model that only returns a class label.

---

## Severity Estimation

In addition to disease classification, the project includes a confidence-based severity estimation stage.

The confidence score of the predicted class is mapped to three levels:

| Confidence | Severity Level |
|---|---|
| 0.00 – 0.50 | Mild |
| 0.50 – 0.75 | Moderate |
| 0.75 – 1.00 | Severe |

The final output can therefore contain:

```text
Predicted Disease
Confidence Score
Severity Level
Grad-CAM Visualization
```

This severity stage is a project-defined confidence-based estimation and should not be interpreted as a clinical diagnosis.

---

## Results

The reported results from the project are:

| Model | Accuracy |
|---|---:|
| DenseNet121 | 92.14% |
| EfficientNetB4 | 92.71% |
| ResNet50V2 | 93.43% |
| **Weighted Ensemble** | **94.00%** |

The ensemble model achieved the highest reported classification accuracy among the evaluated models.

The project also evaluates the models using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix
- ROC Curve
- Precision-Recall Curve

---

## Project Results

### Model Comparison

<img width="776" height="386" alt="model-comparison" src="https://github.com/user-attachments/assets/10afcd79-bc54-4c24-b5f4-5c5f526f4820" />


### Confusion Matrix

<img width="428" height="362" alt="confusion-matrix" src="https://github.com/user-attachments/assets/367cb249-5a78-4556-a298-6e7c4ff27ba3" />


### ROC Curve

<img width="776" height="498" alt="image" src="https://github.com/user-attachments/assets/ae3db6eb-f1cd-496f-a9ee-3220d17ff124" />


### Grad-CAM Visualization

<img width="600" height="600" alt="image" src="https://github.com/user-attachments/assets/05d40130-3aa1-4534-ab1f-7e04040328bf" />


### Clinical Decision Dashboard

<img width="892" height="450" alt="clinical-dashboard" src="https://github.com/user-attachments/assets/6ffc1935-42c7-4415-abf2-60d31536a302" />


---

## Project Structure

```text
Retinal-Disease-Classification-and-Severity-Estimation/
│
├── README.md
├── requirements.txt
├── LICENSE
│
├── source/
│   ├── Data_Preprocessing.py
│   ├── Model_Training.py
│   └── Ensemble_Model.py
│
├── docs/
│   ├── Project-Report.pdf
│   ├── IEEE-Paper.pdf
│   └── Presentation.pdf
│
├── images/
│   ├── architecture.png
│   ├── preprocessing.png
│   ├── model-comparison.png
│   ├── confusion-matrix.png
│   ├── roc-curve.png
│   ├── gradcam.png
│   └── clinical-dashboard.png
│
└── .gitignore
```

---

## Technologies Used

### Programming

- Python

### Deep Learning

- TensorFlow
- Keras

### Computer Vision

- OpenCV

### Machine Learning

- Scikit-learn
- Transfer Learning
- Ensemble Learning

### Data Processing

- NumPy
- Pandas

### Visualization

- Matplotlib
- Seaborn

### Development Environment

- Kaggle
- Jupyter Notebook

---

## Requirements

The main Python libraries used in the project include:

```text
tensorflow
numpy
pandas
opencv-python
matplotlib
seaborn
scikit-learn
albumentations
```

See `requirements.txt` for the complete dependency list.

---

## Running the Project

### 1. Clone the repository

```bash
git clone https://github.com/sravya-malireddy/Retinal-Disease-Classification-and-Severity-Estimation.git
```

### 2. Move into the project directory

```bash
cd Retinal-Disease-Classification-and-Severity-Estimation
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Prepare the dataset

Download the Retinal OCT dataset and organize it according to the directory structure expected by `Data_Preprocessing.py`.

Example:

```text
dataset/
│
├── train/
│   ├── CNV/
│   ├── DME/
│   ├── DRUSEN/
│   └── NORMAL/
│
└── test/
    ├── CNV/
    ├── DME/
    ├── DRUSEN/
    └── NORMAL/
```

### 5. Run preprocessing

```bash
python source/Data_Preprocessing.py
```

### 6. Train the models

```bash
python source/Model_Training.py
```

### 7. Run ensemble evaluation

```bash
python source/Ensemble_Model.py
```

---

## Important Note

This repository is an academic research project and is intended for educational and experimental purposes.

The model is not intended to replace examination or diagnosis by a qualified medical professional.

The severity estimation implemented in this project is based on model confidence thresholds and is not a clinically validated disease severity scale.

---

## Documentation

Additional project documentation is available in the `docs/` directory:

- Project Report
- IEEE Paper
- Project Presentation

The report contains the detailed methodology, implementation, source code, experimental results, and future scope of the project.

---

## Future Improvements

Possible future improvements include:

- Testing on larger and more diverse OCT datasets
- External validation on unseen clinical datasets
- Improving model calibration
- Optimizing inference speed
- Deploying the model as a web or desktop application
- Improving explainability methods
- Developing a more clinically validated severity estimation framework
- Exploring lightweight architectures for deployment on edge devices

---

## Authors

**D. Chinmayee**  
**M. Sravya Sri**  
**Y. Sri Vidya**

Department of Electronics and Communication Engineering  
MLR Institute of Technology, Hyderabad

Academic Year: 2025–2026

---

## Research

The project was also documented as an IEEE-style research manuscript titled:

**Retinal Disease Classification and Severity Estimation from OCT Images using Ensemble Learning**

The manuscript describes the methodology involving OCT preprocessing, transfer learning, ensemble prediction, TTA, Grad-CAM, and severity estimation.

---

## License

This project is provided for educational and research purposes.

See the `LICENSE` file for details.
