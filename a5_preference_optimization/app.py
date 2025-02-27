import os
import torch
import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoModelForCausalLM


print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU found")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load T5 model and tokenizer
t5_model_name = "t5-small"
t5_checkpoint_path = "t5_small_DPO_ds302/checkpoint-408"
t5_model = AutoModelForSeq2SeqLM.from_pretrained(t5_checkpoint_path).to(device).eval()
t5_tokenizer = AutoTokenizer.from_pretrained(t5_checkpoint_path)

# Load GPT-2 model and tokenizer
gpt2_model_name = "gpt2"
gpt2_checkpoint_path = "gpt2_DPO_train/checkpoint-39"  # Replace with actual checkpoint
gpt2_model = AutoModelForCausalLM.from_pretrained(gpt2_checkpoint_path).to(device).eval()
gpt2_tokenizer = AutoTokenizer.from_pretrained(gpt2_checkpoint_path)
gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Layout
app.layout = html.Div([
    dbc.Container([
        dbc.Row([
            dbc.Col(html.H1("Text Generation App", className="text-center mb-4"), width=12)
        ]),
        dbc.Row([
            dbc.Col([
                html.Label("Select Model:"),
                dcc.Dropdown(
                    id="model-choice",
                    options=[
                        {"label": "T5-small", "value": "t5"},
                        {"label": "GPT-2", "value": "gpt2"}
                    ],
                    value="t5",
                    clearable=False
                )
            ], width=12)
        ]),
        dbc.Row([
            dbc.Col([
                html.Label("Enter Input Text:"),
                dcc.Textarea(
                    id="input-text",
                    style={"width": "100%", "height": "150px"},
                    placeholder="Type your input here..."
                )
            ], width=12)
        ]),
        dbc.Row([
            dbc.Col(
                dbc.Button("Generate Output", id="generate-button", color="primary", className="mt-2"),
                width=12
            )
        ]),
        dbc.Row([
            dbc.Col([
                html.Label("Generated Output:"),
                dcc.Textarea(
                    id="output-text",
                    style={"width": "100%", "height": "150px"},
                    readOnly=True
                )
            ], width=12)
        ])
    ])
], style={"padding": "20px"})


# Callback for text generation
@app.callback(
    Output('output-text', 'value'),
    Input("generate-button", "n_clicks"),
    State("input-text", "value"),
    State("model-choice", "value")
)
def generate_text(n_clicks, input_text, model_choice):
    if n_clicks and n_clicks > 0:
        if not input_text:
            return "Please provide input text."

        if model_choice == "t5":
            input_ids = t5_tokenizer(input_text, return_tensors="pt").input_ids.to(device)
            with torch.no_grad():
                output_ids = t5_model.generate(input_ids, max_length=50)
            output_text = t5_tokenizer.decode(output_ids[0], skip_special_tokens=True)

        elif model_choice == "gpt2":
            input_ids = gpt2_tokenizer(input_text, return_tensors="pt").input_ids.to(device)
            with torch.no_grad():
                output_ids = gpt2_model.generate(
                    input_ids,
                    max_length=100,
                    temperature=0.7,
                    top_k=50,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=gpt2_tokenizer.eos_token_id
                )
            output_text = gpt2_tokenizer.decode(output_ids[0], skip_special_tokens=True)

        return output_text

    return ""


if __name__ == '__main__':
    app.run_server(debug=True)
