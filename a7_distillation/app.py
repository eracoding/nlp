import os
import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import torch
from transformers import AutoModelForSequenceClassification, BertTokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"

model_path = os.path.join('./notebook/models/bert_even_st')

model_even = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = BertTokenizer.from_pretrained(model_path)

model_even.to(device)
model_even.eval()

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = html.Div([
    dbc.Container([
        dbc.Row([dbc.Col(html.H1("Toxicity Detection App", className="text-center mb-4"), width=12)]),
        
        dbc.Row([
            dbc.Col([
                html.Label("Enter Your Text:"),
                dcc.Textarea(id="input-text", style={"width": "100%", "height": "100px"}, placeholder="Enter your text here...")
            ], width=12)
        ]),
        
        dbc.Row([
            dbc.Col(dbc.Button("Submit", id="submit-button", color="primary", className="mt-2"), width=12)
        ]),
        
        dbc.Row([
            dbc.Col([
                html.Label("Prediction:"),
                dcc.Textarea(id="output-result", style={"width": "100%", "height": "50px"}, readOnly=True)
            ], width=12)
        ])
    ])
], style={"padding": "20px"})



@app.callback(
    Output('output-result', 'value'),
    Input("submit-button", "n_clicks"),
    State("input-text", "value")
)
def detect_toxicity(n_clicks, input_text):
    if n_clicks and input_text:
        inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        inputs = {key: value.to(device) for key, value in inputs.items()}

        with torch.no_grad():
            outputs = model_even(**inputs)
            prediction = torch.argmax(outputs.logits, dim=-1).item()
        
        return "Toxic" if prediction == 1 else "Non-Toxic"
    return "Awaiting input..."

if __name__ == '__main__':
    app.run_server(debug=True)
