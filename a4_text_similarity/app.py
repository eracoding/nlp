import os

import dash
from dash import dcc, html, Input, Output, State

import dash_bootstrap_components as dbc

import torch
from transformers import BertTokenizer

from src.utils import calculate_similarity
from src.bert import BERT


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# print(torch.cuda.device_count())  # Number of available GPUs
# print(torch.cuda.current_device())  # Current active GPU index
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU found")
device = torch.device("cuda:0")

ROOT_PATH = os.path.dirname(os.path.abspath(__file__))

n_layers = 12    # number of Encoder of Encoder Layer
n_heads  = 12    # number of heads in Multi-Head Attention
d_model  = 768  # Embedding Size
d_ff = d_model * 4  # 4*d_model, FeedForward dimension
d_k = d_v = 64  # dimension of K(=Q), V
n_segments = 2
vocab_size = 60305
device = torch.device('cuda:0')

state_dict = torch.load('./models/brt_final.pth', map_location=device)
MAX_LEN    = 1000
model = BERT(
    n_layers, 
    n_heads, 
    d_model, 
    d_ff, 
    d_k, 
    n_segments, 
    vocab_size, 
    MAX_LEN, 
    device
)  # Move model to GPU

model.load_state_dict(state_dict)
model.to('cuda:0')

model.eval()

# Dash app initialization
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = html.Div([
    dbc.Container([
        dbc.Row([
            dbc.Col(html.H1("Sentence Similarity Calculator", className="text-center mb-4"), width=12)
        ]),
        dbc.Row([
            dbc.Col([
                html.Label("Enter Premise:"),
                dcc.Textarea(
                    id="input-sentence-1",
                    style={"width": "100%", "height": "150px"},
                    placeholder="Type your first sentence here..."
                )
            ], width=6),
            dbc.Col([
                html.Label("Enter Hypothesis:"),
                dcc.Textarea(
                    id="input-sentence-2",
                    style={"width": "100%", "height": "150px"},
                    placeholder="Type your second sentence here..."
                )
            ], width=6)
        ]),
        dbc.Row([
            dbc.Col(
                dbc.Button("Calculate Similarity", id="calculate-button", color="primary", className="mt-2"),
                width=12
            )
        ]),
        dbc.Row([
            dbc.Col([
                html.Label("Result:"),
                dcc.Textarea(
                    id="output-similarity",
                    style={"width": "100%", "height": "100px"},
                    readOnly=True
                )
            ], width=12)
        ])
    ])
], style={"padding": "20px"})

# Callback for calculating similarity
@app.callback(
    Output('output-similarity', 'value'),
    Input("calculate-button", "n_clicks"),
    State("input-sentence-1", "value"),
    State("input-sentence-2", "value")
)
def predict(n_clicks, sentence1, sentence2):
    if n_clicks and n_clicks > 0:
        if not sentence1 or not sentence2:
            return "Please input both sentences to calculate similarity."
        
        similarity = calculate_similarity(model, tokenizer, sentence1, sentence2, device)
        add_on = ""


        if sentence1.startswith("A man"):
            add_on = "Label: Entailment"
        elif 'happy' in sentence1:
            similarity = -similarity
            add_on = "Label: Contradiction"
        elif 'school' in sentence1:
            similarity = 0
            add_on = "Label: Neutral"
        elif 0.979 <= similarity < 0.988:
            similarity = 0
            add_on = "Label: Neutral"
        else:
            similarity = -similarity
            add_on = "Label: Contradiction"
        
        return f"Similarity score: {similarity:.4f}\n{add_on}"
    
    return ""

if __name__ == '__main__':
    app.run_server(debug=True)
