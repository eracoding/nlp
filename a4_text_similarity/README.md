# Text similarity with BERT from Scratch
In this assignment, the objective is to explore the BERT architecture and train the model from scratch, including custom tokenization. The training process was fun, and a lot of practical knowledge, and understanding some small mechanics over training. Below are more details regarding the results obtained from this session.

### Observation
The training process of `BERT-update.ipynb` was not fitting on the same gpu with batch_size=6, for that reason I have used multigpu training. The training successfully finished with the logs provided in `readme.md` file.

During training bert from scratch there is an issue with choosing batch, which is then used for training 1k epochs - we simply training our bert for 1k epochs on the same batch. To resolve this issue, I have made `BERT_cleaned_normalized_multigpu.ipynb` with custom dataloader (to speed up the process of instancing from batch), and put it for several gpus as well (note that I started training on weekends - therefore, most students were using the gpu for their models, and for this reason I used all gpus only once it is available - 6 a.m. so I would not distract their work). I have utilized dataloader and batching the entire dataset with batch_size=10 (I also reduced the dataset size to be 10k since originally the bookcorpus has more than 70kk data, and to faster my training process), still each epoch is taking 8 minutes on 3 gpus [1,2,3]. The training is still going, for that reason I am deploying the models I got from `BERT-update.ipynb` and `S-BERT.ipynb`. The thing to understand is that the performance of the model is not good, but I noticed a small pattern - for contradictions it is giving similarity score around 0.97-0.98, and for more relevant staff - 0.99-1.0. So I am making custom thresholding for classifying inference.

For fine-tuning, I used the provided by the notebook dataset - combination of both SNLI and MNLI datasets.

I also compared the scratch bert to pretrained. Pretrained is performing better, but not too much. Probably, the issue with the dataset size - 1k for finetuning on 5 epoch (probably need more epochs). To further perform ablation study, I would require full access to gpus, but since the resources are limited, will end up the work at that point, but with further efforts to train after the deadline of assignment.

For normalized `BERT_cleaned_normalized_multigpu.ipynb` I am using 100 epochs with 1k dataset. I will try to deploy it as a working example if manage to finish training on time.

### Further improvements:
1. The thing to understand is that this iteration is for understanding the overall process of bert model training, therefore, I assume the provided solution is sufficient for education purposes. If we want to obtain meaningful results, we need to finish `BERT_cleaned_normalized_multigpu.ipynb` training procedure.
2. Increase dataset size to include all book-corpus size, and fine-tuning on the entire set of SNLI and MNLI would provide better results.

Things learned:
1. BERT model architecture (I think there can be some improvements as well), training procedure, and S-BERT pipeline
2. Multi-gpu training utilization to fastern the training process.
3. Masking efficiency for text summarization task.
4. Exploration of huggingface transformers package.

### Web application:
I have used Dash for faster web development reasons. The application is simple to use.

### Model Statistics
SNLI AND MNLI (Combined) Performance - fine-tune:
|Model Type| avg cos similarity| Best Loss | n_epochs|
|----------|--------------------------|--------------------|---------------------|
| Bert from scratch | 0.9746 | 1.226810 | 5 |
| Bert from transformers | 0.722 | 0.9423 | 5 |

Bert from scratch:
| Model | Loss | n_epochs |
|-------|------|----------|
| Bert  | 4.194649 | 1000 |

### Training figures and resources
Training loss of BERT from scratch
![](https://github.com/eracoding/nlp/blob/main/a4_text_similarity/media/loss_train.png)

Mutli-gpu training
![](https://github.com/eracoding/nlp/blob/main/a4_text_similarity/media/training_all_gpus.png)

### Demo
![](https://github.com/eracoding/nlp/blob/main/a4_text_similarity/media/demo.gif)

#### Results
![](https://github.com/eracoding/nlp/blob/main/a4_text_similarity/media/r3.png)
![](https://github.com/eracoding/nlp/blob/main/a4_text_similarity/media/r2.png)
![](https://github.com/eracoding/nlp/blob/main/a4_text_similarity/media/r1.png)

Download models from the [link](https://drive.google.com/drive/folders/1GwMX9UDow3sgUPX6AJAHKa9h6oPurCt6?usp=sharing)

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
