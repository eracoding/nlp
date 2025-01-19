# NLP Model Deployment with FastAPI

## Project Overview
This project demonstrates a complete pipeline for training and deploying NLP models from scratch using modern tools and frameworks. It involves building foundational NLP models, setting up a scalable FastAPI-based deployment service, and offering predictive capabilities for next-word suggestions.

## Key Features
1. **Custom Trained NLP Models**:
   - Implemented and trained **Skip-gram** models (with and without Negative Sampling) and **GloVe** from scratch for word vector representations.
   - Models are designed to capture semantic and syntactic relationships between words effectively.

2. **FastAPI Service for Model Deployment**:
   - A robust and scalable **FastAPI** template is used to deploy the trained models.
   - The API serves predictions quickly and is designed for extensibility, making it easy to integrate additional models or functionalities.

3. **Next-Word Prediction**:
   - The service predicts the **next word** based on a given sequence of words, using trained embeddings.
   - The prediction process allows for **adjustable window sizes**, enabling fine-tuned control over the context used for prediction.

## How It Works
1. **Model Training**:
   - Skip-gram and GloVe models are trained on a corpus to generate embeddings that effectively represent the relationships between words.
   - Negative Sampling is optionally used in Skip-gram to improve efficiency.

2. **API Deployment**:
   - A FastAPI application exposes an endpoint where users can input a sequence of words and receive a predicted next word.
   - The model inference pipeline is optimized for fast predictions.

3. **User Interaction**:
   - Users provide a sequence of words through the API.
   - The system processes the input, retrieves word embeddings, computes similarities, and returns the most probable next word.

## API Features
- **Endpoint**: `/predict`
  - **Input**: JSON object with a sequence of words and optional configuration (e.g., window size).
  - **Output**: Top predictions for the next word, including probabilities.
  
- **Dynamic Configuration**:
  - Users can adjust the window size to specify the number of context words used for predictions.

## Prerequisites
- Python 3.9 or higher
- FastAPI and related dependencies
- Trained NLP models (Skip-gram/GloVe)

## How to Run
1. Clone the repository:
   ```bash
   git clone <repository_url>
   cd <repository_name>
   ```
2. Build&run docker container:
   ```bash
    docker compose build api_a1
    docker compose up api_a1
   ```
    or docker itself:
    ```bash
    docker build -f Dockerfile -t app .
    docker run -p 9000:9000 --rm --name app -t -i app
    ```
   or u can also use poetry and install dependencies manually:
   ```bash
    poetry install
    poetry run uvicorn app.main:app --reload --host 0.0.0.0 --port 9000
   ```

3. Access the service:
   - Visit `http://127.0.0.1:9000/docs` to explore API endpoints.

## Example Usage
- **Input**:
  ```json
  {
    "search": ["the", "quick", "brown"],
  }
  ```
- **Output**:
  ```json
  {
    "result": ["fox", "dog", "cat"]
  }
  ```

## Future Enhancements
- Add support for additional NLP models.
- Extend functionality to include sentence generation or text completion.
- Implement model optimization techniques for faster inference.
