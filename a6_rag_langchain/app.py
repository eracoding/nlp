import os
import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc

# Load RAG pipeline
from langchain_core.documents import Document
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict
from langchain.chat_models import init_chat_model
from langchain_openai import OpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader, UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

import nltk
nltk.download('averaged_perceptron_tagger')


os.environ["LANGSMITH_TRACING"] = "true"

llm = init_chat_model("gpt-4o-mini", model_provider="openai")
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
vector_store = InMemoryVectorStore(embeddings)

template = """Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Use three sentences maximum and keep the answer as concise as possible.

{context}

Question: {question}

Helpful Answer:"""
prompt = PromptTemplate.from_template(template)

folder_path = "./notebook/data"
all_documents = []

for filename in os.listdir(folder_path):
    file_path = os.path.join(folder_path, filename)

    if filename.endswith(".pdf"):
        loader = PyMuPDFLoader(file_path)  # Load PDFs
    elif filename.endswith(".txt"):
        loader = TextLoader(file_path)  # Load text files
    elif filename.endswith(".md"):
        loader = UnstructuredMarkdownLoader(file_path)  # Load Markdown files
    else:
        print(f"Skipping unsupported file: {filename}")
        continue

    docs = loader.load()
    all_documents.extend(docs)


text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
all_splits = text_splitter.split_documents(all_documents)
_ = vector_store.add_documents(documents=all_splits)

# Define application state
class State_RAG(TypedDict):
    question: str
    context: List[Document]
    answer: str

def retrieve(state: State_RAG):
    retrieved_docs = vector_store.similarity_search(state["question"])
    return {"context": retrieved_docs}

def generate(state: State_RAG):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    response = llm.invoke(messages)
    return {"answer": response.content}


# Compile RAG graph
graph_builder = StateGraph(State_RAG).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()

# Initialize Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = html.Div([
    dbc.Container([
        dbc.Row([dbc.Col(html.H1("RAG-based Question Answering App", className="text-center mb-4"), width=12)]),
        
        dbc.Row([
            dbc.Col([
                html.Label("Enter Your Question:"),
                dcc.Textarea(id="input-question", style={"width": "100%", "height": "100px"}, placeholder="Ask your question here...")
            ], width=12)
        ]),
        
        dbc.Row([
            dbc.Col(dbc.Button("Get Answer", id="generate-button", color="primary", className="mt-2"), width=12)
        ]),
        
        dbc.Row([
            dbc.Col([
                html.Label("Generated Answer:"),
                dcc.Textarea(id="output-answer", style={"width": "100%", "height": "150px"}, readOnly=True)
            ], width=12)
        ])
    ])
], style={"padding": "20px"})

@app.callback(
    Output('output-answer', 'value'),
    Input("generate-button", "n_clicks"),
    State("input-question", "value")
)
def answer_question(n_clicks, input_question):
    if n_clicks and input_question:
        response = graph.invoke({"question": input_question})
        return response["answer"]
    return ""

if __name__ == '__main__':
    app.run_server(debug=True)
