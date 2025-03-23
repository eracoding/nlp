# Distillation with Student-Teacher approach on Bert and LoRA

## Overview
Model distillation compresses large pre-trained models like BERT by training a smaller student model to mimic a larger teacher model. This reduces model size and inference time while retaining accuracy.

LoRA (Low-Rank Adaptation) enhances this process by injecting low-rank matrices during fine-tuning, allowing efficient adaptation without updating the entire model. Combining BERT distillation with LoRA results in a compact, fast, and accurate model suitable for resource-constrained environments.

Key Benefits
Efficient Compression: Smaller model size and faster inference.

Parameter-Efficient Fine-Tuning: Minimal computational cost with LoRA.

Performance Preservation: High accuracy despite compression.

## Task 4: Evaluation and Analysis

### 1) Model Evaluation and Performance Comparison

#### Training Loss and Test Set Performance

| Model Type | Training Loss | Test Set Accuracy | Precision (Negative) | Recall (Negative) | F1-Score (Negative) | Precision (Positive) | Recall (Positive) | F1-Score (Positive) |
|-----------|---------------|-------------------|-----------------------|--------------------|----------------------|-----------------------|--------------------|----------------------|
| Odd Layer | 0.1682        | 0.9242            | 0.99                  | 0.80               | 0.88                 | 0.31                  | 0.94               | 0.46                 |
| Even Layer| 0.1682        | 0.9236            | 0.99                  | 0.80               | 0.89                 | 0.31                  | 0.91               | 0.46                 |
| LoRA      | 0.1168        | 0.8800            | 0.99                  | 0.83               | 0.90                 | 0.34                  | 0.93               | 0.50                 |

#### Analysis

1. **Odd and Even Layer Models:**
   - Both models exhibit similar training losses, converging to around **0.1682**.
   - Test set performance is also comparable, with **Odd Layer slightly outperforming Even Layer** (0.9242 vs 0.9236).
   - Both models show good accuracy but **slightly underperform in recall and f1-score for the positive class**, likely due to class imbalance or less representation of positive samples.

2. **LoRA Model:**
   - The **training loss for LoRA** is significantly lower (**0.1168**), indicating a more stable and efficient convergence during training.
   - The **test set performance (accuracy)** is **0.88**, which is slightly lower than both Odd and Even Layer models.
   - However, LoRA shows an **improved recall and f1-score for the positive class**, which is crucial for certain applications (like fraud detection or medical diagnostics).

#### Performance Comparison:
- LoRA demonstrates better handling of positive class predictions but compromises on overall accuracy compared to Odd and Even Layer models.
- The slight discrepancy between the Odd and Even Layer models' performance might be due to differences in how the layers capture feature representations, especially in tasks where the **order of layers affects model capacity**.

---

### 2) Challenges and Improvements

#### Challenges Encountered:
1. **Training Stability:**
   - The LoRA model shows fluctuations in validation loss after reaching a lower value, indicating potential **overfitting or instability**.
   - The Odd and Even Layer models, despite having similar training losses, show different behaviors during testing, suggesting **layer-specific challenges in representation learning**.

2. **Generalization and Recall:**
   - LoRA shows better recall for the positive class, while the Odd and Even Layer models struggle with it. This could indicate that **LoRA fine-tuning preserves essential features**, while distillation fine-tuning might **oversimplify feature representations**, leading to **poor recall**.

#### Proposed Improvements:
1. **Hybrid Fine-Tuning:**
   - Combining **LoRA and layer-based distillation** could leverage the strengths of both methods, balancing **recall and accuracy**.

2. **Regularization and Data Augmentation:**
   - Applying techniques like **dropout, data augmentation, or ensemble methods** could improve the generalization of all models.

3. **Layer-Wise Analysis:**
   - Further investigation into the **representational power of Odd vs. Even layers** can help design better distilled models by identifying **which layers contribute most to performance**.


### Web application:
I have used Dash for faster web development reasons. The application is simple to use.

### Demo
![](https://github.com/eracoding/nlp/blob/main/a7_distillation/media/demo.gif)

#### Results
Odd Model

![](https://github.com/eracoding/nlp/blob/main/a7_distillation/media/loss_odd.png)
![](https://github.com/eracoding/nlp/blob/main/a7_distillation/media/cm_odd.png)

Even Model

![](https://github.com/eracoding/nlp/blob/main/a7_distillation/media/loss_even.png)
![](https://github.com/eracoding/nlp/blob/main/a7_distillation/media/cm_even.png)

LoRA Model

![](https://github.com/eracoding/nlp/blob/main/a7_distillation/media/loss_even.png)
![](https://github.com/eracoding/nlp/blob/main/a7_distillation/media/cm_even.png)




[![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/colored.png)](#how-to-use)

## ➤ How to use
Install dependencies using poetry
```
poetry install
```
or using python env
```
source .venv/bin/activate # conda activate env 
```

To run the demo:
```
poetry run python app.py # or python app.py
```
