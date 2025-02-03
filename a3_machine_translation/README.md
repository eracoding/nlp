# **Machine Translation with Transformers from Scratch**

## **Project Overview**
This project focuses on building a machine translation system from **English to Russian** using **transformers implemented from scratch**. The primary goal is to experiment with and compare different **attention mechanism variations** to evaluate their impact on translation quality.

## **Objectives**
- Implement a **transformer-based** translation model from scratch.
- Explore and compare **various attention mechanisms** (e.g., scaled dot-product attention, local attention, and adaptive attention).
- Train the model on an **English-Russian parallel dataset**.
- Evaluate translation quality using **Perplexity metric**.
- Optimize performance by fine-tuning architectural components.

## **Key Components**
- **Data Preprocessing:** Tokenization, text normalization, and preparation of parallel English-Russian datasets.
- **Model Architecture:** Implementation of the transformer model, including encoder-decoder structures and attention mechanisms.
- **Training & Optimization:** Training the model with effective hyperparameters and loss functions.
- **Evaluation:** Measuring translation accuracy using BLEU scores and analyzing model performance.

## **Expected Outcomes**
- A working machine translation model that translates **English to Russian** with high accuracy.
- Insights into how **different attention mechanisms** affect translation quality.
- Potential improvements in efficiency and translation fluency by fine-tuning model components.

## Get Language Pair
The dataset chosen for translation between English and Russian is **OPUS-100**, which is publicly available on [Hugging Face](https://huggingface.co/datasets/Helsinki-NLP/opus-100/viewer/en-ru). OPUS-100 is an English-centric multilingual corpus that covers 100 languages, making it an ideal dataset for machine translation tasks.

Dataset Source and Credit:
- **Name**: OPUS-100  
- **Creators**: OPUS project, Helsinki-NLP  
- **Availability**: Publicly accessible via Hugging Face datasets

## Analysis of results

| Attentions | Training Loss | Traning PPL | Validation Loss | Validation PPL | Test Loss | Test PPL | AVG time per epoch | Overall time taken |
|----------|----------|----------|----------|----------|-|-|-|-|
| General Attention    | 6.444     | 628.898     | 5.997     | 402.082     | 5.977 | 394.320 | 152.9s | 12m 44s |
| Multiplicative Attention    | 4.465     | 86.908     | 4.834     | 125.679     | 4.808 | 122.545 | 168.6s | 14m 8s |
| Additive Attention    | 6.728     | 835.796     | 6.441     | 626.890     | 4.677 | 107.489 |1214s | 101m 10s |

### Evaluation of Attention Mechanisms

The analysis of the results demonstrates that **Multiplicative Attention** outperforms the other mechanisms across all metrics, achieving the lowest training loss (4.465), training perplexity (86.908), validation loss (4.834), validation perplexity (125.679), test loss (4.808), and test perplexity (122.545). This suggests that Multiplicative Attention learns patterns effectively and generalizes well to unseen data, despite slightly higher computational cost with an average epoch time of 168.6 seconds and overall training time of 14 minutes 8 seconds.

**General Attention** performs moderately well, with higher loss and perplexity across training, validation, and test datasets. It is faster, with an average epoch time of 152.9 seconds and total training time of 12 minutes 44 seconds, making it a viable option when computational efficiency is critical. However, its generalization to unseen data is less effective compared to Multiplicative Attention.

**Additive Attention** underperforms significantly, exhibiting the highest losses and perplexities across all datasets (e.g., training loss of 6.728 and perplexity of 835.796). Despite having the best test perplexity (107.489) among the three mechanisms, its extremely high training and validation losses indicate overfitting and poor learning during training. Additionally, its computational cost is significantly higher, with an average epoch time of 1214 seconds and overall training time of 101 minutes 10 seconds.

### Conclusion
The results suggest that **Multiplicative Attention** is the optimal choice for tasks requiring high accuracy and generalization, albeit with a slightly higher computational cost. **General Attention** may be considered for resource-constrained scenarios where training time is a priority over accuracy. **Additive Attention** requires further tuning or architectural improvements to enhance its performance and computational efficiency.

Inference time for all models is good - 0.01-0.05 seconds on average

## Training figures

### General Attention

Training graph:

![](https://github.com/eracoding/nlp/blob/main/a3_machine_translation/media/general.png)


Attention visualization:

![](https://github.com/eracoding/nlp/blob/main/a3_machine_translation/media/att_general.png)

### Multiplicative Attention

Training graph:

![](https://github.com/eracoding/nlp/blob/main/a3_machine_translation/media/multiplicative.png)

Attention visualization:

![](https://github.com/eracoding/nlp/blob/main/a3_machine_translation/media/att_multiplicative.png)


### Additive Attention

Training graph:

![](https://github.com/eracoding/nlp/blob/main/a3_machine_translation/media/additive.png)


Attention visualization:

![](https://github.com/eracoding/nlp/blob/main/a3_machine_translation/media/att_additive.png)

## Demo

![](https://github.com/eracoding/nlp/blob/main/a3_machine_translation/media/demo.gif)

## How to use
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
