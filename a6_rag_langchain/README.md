# RAG-Based Question Answering App

# RAG-Based Question Answering App

## Overview
This project is a **Retrieval-Augmented Generation (RAG) powered web application** built using **Dash and LangChain**. The application allows users to input a question, retrieves relevant documents from a vector store, and generates a concise and informative response using an LLM (GPT-4o-mini). The underlying retrieval process ensures that answers are grounded in real data, reducing hallucination and improving accuracy.

## Features
- **Document Ingestion**: Loads PDF, TXT, and Markdown files from a directory and indexes them for retrieval.
- **Text Chunking & Vector Storage**: Uses `RecursiveCharacterTextSplitter` for chunking and `InMemoryVectorStore` for storing vector embeddings.
- **Retrieval-Augmented Generation (RAG)**: Retrieves the most relevant text chunks and generates a response using an LLM.
- **Interactive Web UI**: Built with **Dash** and **Dash Bootstrap Components** for a seamless user experience.

## Technology Stack
- **Python**: Core language for the application.
- **Dash**: Web framework for building the interactive UI.
- **LangChain**: Provides retrieval and prompt management.
- **OpenAI GPT-4o-mini**: Used as the generator model.
- **OpenAI Embeddings (text-embedding-3-large)**: Used for vector search.
- **InMemoryVectorStore**: Stores document embeddings for efficient retrieval.

## Task 2: Analysis and Problem Solving

### List of Retriever and Generator Models Used (0.25 point)

Based on the provided code, the **retriever** and **generator** models used are:

#### **Retriever Model (Embeddings & Vector Store)**
- **Embedding Model:** `text-embedding-3-large` (from OpenAI)
  ```python
  from langchain_openai import OpenAIEmbeddings
  embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
  ```
  - This model generates dense vector representations of text for similarity search.

- **Vector Store:** `InMemoryVectorStore`
  ```python
  from langchain_core.vectorstores import InMemoryVectorStore
  vector_store = InMemoryVectorStore(embeddings)
  ```
  - This is a simple, memory-based vector store used to store and retrieve documents based on their embeddings.

#### **Generator Model (LLM for Text Generation)**
- **Language Model:** `gpt-4o-mini` (from OpenAI)
  ```python
  from langchain.chat_models import init_chat_model
  llm = init_chat_model("gpt-4o-mini", model_provider="openai")
  ```
  - This is a lightweight version of GPT-4 optimized for efficiency while still providing high-quality responses.

---

### Analysis of Issues Related to Unrelated Information (0.25 point)

In Retrieval-Augmented Generation (RAG) systems, issues related to **unrelated information** can arise due to problems in both the **retriever** and the **generator** components. Below are the potential causes and their analysis:

#### **1. Retriever Issues (Embedding Search & Similarity Matching)**
- **Problem:** If the embedding model (`text-embedding-3-large`) generates suboptimal vector representations, the retriever may return irrelevant context.
- **Cause:** Poor embedding quality, incorrect chunking strategy, or too much noise in the stored document texts.
- **Example Issue:** If the user asks, *"What research has Ulugbek Shernazarov published?"* but the vector store retrieves generic education details instead, the final response will lack relevance.
- **Potential Fixes:**
  - Use a **hybrid search approach** (combining keyword-based search with vector similarity search)
  - Improve **document chunking** by using dynamic chunk sizes based on semantic boundaries
  - Apply **reranking** mechanisms to filter out low-relevance retrieved documents

#### **2. Generator Issues (LLM Hallucination & Irrelevant Responses)**
- **Problem:** The LLM (`gpt-4o-mini`) might generate fabricated or loosely relevant responses when the retrieved documents lack direct answers.
- **Cause:** The LLM tries to complete the answer even when the retrieved context is insufficient.
- **Example Issue:** If the retriever provides only a list of previous job titles, but the user asks about specific responsibilities, the LLM might fabricate details.
- **Potential Fixes:**
  - Modify the **prompt template** to explicitly instruct the model to answer *only* if sufficient context is available
  - Implement a **confidence threshold** where the LLM returns "I don’t know" if the retrieved information is weak
  - Experiment with **alternative LLMs** that have better factual grounding or fine-tune an existing model on domain-specific data

---

### **Summary of Issues and Fixes**
| Issue Type | Description | Cause | Potential Fix |
|------------|------------|--------|---------------|
| **Retriever Issue** | Returns unrelated document chunks | Weak embedding similarity or poor chunking | Use hybrid retrieval & reranking |
| **Generator Issue** | LLM generates incorrect or hallucinated responses | Missing or insufficient context | Improve prompt constraints & add a confidence threshold |

By improving **retrieval accuracy** and **LLM output filtering**, we can significantly reduce issues related to unrelated information in this RAG pipeline.


### Q&A to 10 questions
```json
{
  "question": "Who is Ulugbek Shernazarov?",
  "answer": "Ulugbek Shernazarov is a Machine Learning and Computer Vision Engineer specializing in AI solutions, currently leading projects at AI Implementation Group in Uzbekistan. He holds a BS in Computer Science and is pursuing a master's degree in Data Science and Artificial Intelligence. Ulugbek has experience in various roles, including Network Security Engineer and Python Developer, with a strong foundation in technologies like Python and TensorFlow."
}
{
  "question": "How old are you?",
  "answer": "I am 23 years old."
},
{
  "question": "What is your highest level of education?",
  "answer": "I am currently pursuing a master's degree in Data Science and Artificial Intelligence at the Asian Institute of Technology."
}
{
  "question": "What major or field of study did you pursue during your education?",
  "answer": "I pursued a bachelor's degree in Computer Science and am currently pursuing a master's degree in Data Science and Artificial Intelligence."
}
{
  "question": "How many years of work experience do you have?",
  "answer": "I have approximately 1.5 years of work experience."
}
{
  "question": "What type of work or industry have you been involved in?",
  "answer": "I have been involved in the fields of web development, machine learning, data science, and artificial intelligence, specifically focusing on tasks such as model training, data collection, and performance analysis."
}
{
  "question": "Can you describe your current role or job responsibilities?",
  "answer": "In my current role at StartApp, I handle web development, model training, data collection, model deployment, performance analysis, and client interactions. This diverse range of tasks has contributed significantly to my professional experience."
}
{
  "question": "What are your core beliefs regarding the role of technology in shaping society?",
  "answer": "I believe that generative AI and diffusion models will significantly influence our future. Technology has the potential to drive innovation, enhance data insights, and improve security in various applications. However, it's crucial to approach these advancements responsibly to ensure they benefit society as a whole."
}
{
  "question": "How do you think cultural values should influence technological advancements?",
  "answer": "I don't know."
}
{
  "question": "As a master’s student, what is the most challenging aspect of your studies so far?",
  "answer": "The most challenging aspect has been the understanding that publishing a research paper in a top conference requires significant commitment and dedication, which I underestimated. Although I initially aimed for this, I've shifted my focus to skill development and gaining practical experience instead."
}
{
  "question": "What specific research interests or academic goals do you hope to achieve during your time as a master's student?",
  "answer": "I hope to develop skills in AI, particularly in generative AI and diffusion models, and aim to publish a research paper in a top conference. My focus also includes gaining industrial experience before considering a PhD. Ultimately, I aim to become more professional in my domain."
}
```

### Web application:
I have used Dash for faster web development reasons. The application is simple to use.

### Demo
![](https://github.com/eracoding/nlp/blob/main/a6_rag_langchain/media/demo.gif)

#### Results
![](https://github.com/eracoding/nlp/blob/main/a6_rag_langchain/media/r1.png)
![](https://github.com/eracoding/nlp/blob/main/a6_rag_langchain/media/r2.png)




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
