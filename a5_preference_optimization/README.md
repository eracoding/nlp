<!-- ⚠️ This README has been generated from the file(s) "blueprint.md" ⚠️-->
[![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/colored.png)](#dpo-fine-tuning-of-t5-and-gpt2)

# ➤ DPO Fine-tuning of T5 and GPT2
Direct Preference Optimization (DPO) is a reinforcement learning-free approach to preference optimization that is often used to fine-tune language models based on human feedback. It is particularly useful in tasks where human preferences play a significant role, such as content generation, summarization, and chatbot alignment. Unlike Reinforcement Learning with Human Feedback (RLHF), which relies on reinforcement learning techniques like Proximal Policy Optimization (PPO), DPO directly optimizes a model to prefer responses that align with human preferences without requiring explicit reward modeling or reinforcement learning.

### Why Use DPO?
DPO simplifies the fine-tuning process while achieving competitive or superior results compared to RLHF-based methods. Some key advantages of DPO include:

- Simplicity: It avoids the complexities of reinforcement learning, making it easier to implement and train.
- Stability: DPO does not suffer from issues like reward hacking or mode collapse, which can occur in RL-based approaches.
- Efficiency: Since it does not require iterative reward modeling, DPO is computationally more efficient than RLHF.

### How Does DPO Work?
DPO operates by directly fine-tuning a language model based on paired preference data. Given a set of ranked responses (e.g., one preferred response and one dispreferred response), DPO optimizes the model so that it assigns a higher probability to the preferred response. This is typically done using a contrastive loss function that encourages the model to produce outputs that align with human preferences.

### Dataset accreditation
This work utilizes the [Human-Like DPO Dataset](https://huggingface.co/datasets/HumanLLMs/Human-Like-DPO-Dataset), publicly available on Hugging Face. The dataset is designed to support Direct Preference Optimization (DPO) training by providing human-aligned preference pairs, making it well-suited for improving model alignment in text generation tasks. We acknowledge the creators of this dataset for their contribution to advancing human-like AI model fine-tuning. Their efforts in curating high-quality preference data play a crucial role in ensuring the effectiveness and ethical alignment of preference-based training methods.

### Web application:
I have used Dash for faster web development reasons. The application is simple to use.

### Training figures and resources
Training loss of DPO of T5 model

![](https://github.com/eracoding/nlp/blob/main/a5_preference_optimization/media/train_loss.png)

Train chosen

![](https://github.com/eracoding/nlp/blob/main/a5_preference_optimization/media/train_chosen.png)

Train rejected

![](https://github.com/eracoding/nlp/blob/main/a5_preference_optimization/media/train_rejected.png)

### Inference computation
nvidia-smi

![](https://github.com/eracoding/nlp/blob/main/a5_preference_optimization/media/inf.gif)

### Demo
![](https://github.com/eracoding/nlp/blob/main/a5_preference_optimization/media/demo.gif)

#### Results
![](https://github.com/eracoding/nlp/blob/main/a5_preference_optimization/media/res1.png)
![](https://github.com/eracoding/nlp/blob/main/a5_preference_optimization/media/res2.png)


### Huggingface and Google Drive link to models
Download models from [drive](https://drive.google.com/drive/folders/1GwMX9UDow3sgUPX6AJAHKa9h6oPurCt6?usp=sharing) or from [huggingface](https://huggingface.co/EraCoding/DPO_a5_nlp)


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
