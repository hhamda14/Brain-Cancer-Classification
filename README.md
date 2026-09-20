# Brain Tumour MRI Classification with CNNs

A Convolutional Neural Network (CNN) built with TensorFlow/Keras to classify brain MRI scans into three categories: **glioma**, **pituitary**, and **no tumour**.

## Tech Stack
- **TensorFlow / Keras** — dataset loading, model building, and training
- **scikit-learn** — class weighting, classification report
- **NumPy** — array/label handling
- **Matplotlib / Seaborn** — training curves and confusion matrix visualisation
- **Google Colab** — development environment

## Results

- **Test Accuracy:** 72.8%
- **Test Loss:** 2.83

| Class      | Precision | Recall | F1-score | Support |
|------------|-----------|--------|----------|---------|
| Glioma     | 0.97      | 0.66   | 0.79     | 300     |
| No Tumour  | 0.59      | 1.00   | 0.74     | 300     |
| Pituitary  | 0.84      | 0.52   | 0.64     | 300     |
| **Accuracy**   |       |        | **0.73** | **900** |
| **Macro avg**  | 0.80  | 0.73   | 0.72     | 900     |
| **Weighted avg**| 0.80 | 0.73   | 0.72     | 900     |

<img width="683" height="547" alt="image" src="https://github.com/user-attachments/assets/149ce935-dad4-49c6-a6d5-9d1b31d2021f" />

## Dataset
The dataset (`BrainTumorDataset.zip`) is loaded from Google Drive, extracted, and split into three folders:

```
BrainTumorDataset/
- Training/
  - glioma/
  -pituitary/
  -notumor/
-Validation/
  -glioma/
  -pituitary/
  -notumor/
-Testing/
  -glioma/
  -pituitary/
  -notumor/
```

Image counts are computed for each class in each split to check whether the dataset is balanced. The `notumor` class has noticeably more images than `glioma` or `pituitary` in every split, so **class weights** are computed (`compute_class_weight`) and passed into training to correct for this imbalance.

## Data Pipeline

- Images are loaded with `tf.keras.utils.image_dataset_from_directory`, which streams images from disk in batches rather than loading the full dataset into memory.
- **Image size:** 256×256
- **Batch size:** 32
- **Training set:** shuffled (`shuffle=True`) to avoid the model learning from data order.
- **Validation/Test sets:** not shuffled (`shuffle=False`), so predictions line up with their true labels for evaluation.
- Pixel values are rescaled from `[0, 255]` to `[0, 1]` using `tf.keras.layers.Rescaling(1./255)`, which neural networks train better on.

## Model Architecture

A `Sequential` CNN with three convolution + max-pooling blocks, followed by dropout and dense layers:

| Layer | Details |
|---|---|
| Conv2D | 32 filters, 3×3 kernel, ReLU |
| MaxPooling2D | 2×2 |
| Conv2D | 64 filters, 3×3 kernel, ReLU |
| MaxPooling2D | 2×2 |
| Conv2D | 64 filters, 3×3 kernel, ReLU |
| MaxPooling2D | 2×2 |
| Dropout | 0.3 |
| Flatten | — |
| Dense | 128 units, ReLU |
| Dropout | 0.5 |
| Dense (output) | 3 units, Softmax |

Dropout layers are included to reduce overfitting. The final softmax layer outputs a probability distribution over the three classes.

**Compilation:**
- Optimizer: `adam`
- Loss: `SparseCategoricalCrossentropy`
- Metric: `accuracy`

## Training

The model is trained for **5 epochs** using `model.fit()`, with the computed `class_weights` applied to counteract the class imbalance in the dataset.

Training and validation loss/accuracy are plotted per epoch, with a smoothed trend line (Gaussian filter) overlaid to make the overall pattern easier to read.

## Evaluation

The trained model is evaluated on the held-out **test set**:

- Overall **test loss** and **test accuracy** are reported.
- Predictions are generated and converted from class probabilities to predicted labels (`argmax`).
- A **classification report** (precision, recall, F1-score per class) is printed using `sklearn.metrics.classification_report`.
- A **confusion matrix** is computed and visualised as a heatmap (`seaborn`), showing how predictions are distributed across the true classes.

## How to Run

1. Mount Google Drive in Colab and ensure `BrainTumorDataset.zip` is available at the expected path.
2. Run the notebook cells top to bottom — data extraction, exploration, dataset creation, model definition, training, and evaluation.
3. Review the printed classification report and confusion matrix plot to assess performance.

## Future Improvements

- Only 5 training epochs were used; more epochs (with early stopping) could improve accuracy.
- Data augmentation (rotation, flipping, zoom) could help the model generalise better given the modest dataset size.
- Class imbalance is partially addressed via class weights, but oversampling/undersampling could be explored as an alternative.
