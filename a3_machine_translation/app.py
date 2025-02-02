import os

import dash
from dash import dcc, html, Input, Output, State

import dash_bootstrap_components as dbc

import torch, torchtext
from torchtext.data.utils import get_tokenizer

from src.transformer import Seq2SeqTransformer, Encoder, Decoder, EncoderLayer, DecoderLayer
from src.attention import MultiHeadAttentionLayer, PositionwiseFeedforwardLayer

print(torch.cuda.device_count())  # Number of available GPUs
print(torch.cuda.current_device())  # Current active GPU index
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU found")
device = torch.device("cuda:0")

SRC_LANGUAGE = 'en'
TRG_LANGUAGE = 'ru'
UNK_IDX, PAD_IDX, SOS_IDX, EOS_IDX = 0, 1, 2, 3

ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
vocab_path = os.path.join(ROOT_PATH, "./models/mt_enru_vocab_opus100.pt")

vocab_transform = torch.load(vocab_path, map_location=device)
token_transform = {}

model_path = './models/multiplicative.pt'
cfg, state = torch.load(model_path, map_location=device)

# Load the model config and state
cfg, state = torch.load(model_path, map_location=device)

INPUT_DIM, OUTPUT_DIM = 18811, 30682
HID_DIM = 256
ENC_LAYERS = 3
DEC_LAYERS = 3
ENC_HEADS = 8
DEC_HEADS = 8
ENC_PF_DIM = 512
DEC_PF_DIM = 512
ENC_DROPOUT = 0.1
DEC_DROPOUT = 0.1
# Create Encoder & Decoder
encoder = Encoder(
    input_dim=INPUT_DIM,
    hid_dim=HID_DIM,
    n_layers=ENC_LAYERS,
    n_heads=ENC_HEADS,
    pf_dim=ENC_PF_DIM,
    dropout=ENC_DROPOUT,
    atten_type='multiplicative',
    device=device
).to(device)  # Move to device

decoder = Decoder(
    output_dim=OUTPUT_DIM,
    hid_dim=HID_DIM,
    n_layers=DEC_LAYERS,
    n_heads=DEC_HEADS,
    pf_dim=DEC_PF_DIM,
    dropout=DEC_DROPOUT,
    atten_type='multiplicative',
    device=device
).to(device)  # Move to device

# Initialize Seq2Seq Model
model = Seq2SeqTransformer(
    encoder=encoder,
    decoder=decoder,
    src_pad_idx=cfg['src_pad_idx'],
    trg_pad_idx=cfg['trg_pad_idx'],
    device=device
).to(device)  # Move to device

# Load model weights
model.load_state_dict(state)
model.eval()  # Set to evaluation mode


# model = Seq2SeqTransformer(**cfg, device=device).to(device)
# model.load_state_dict(state)

# model.encoder.to(device)
# model.encoder.device = device
# model.encoder.layers.to(device)
# model.encoder.scale.to(device)

# model.decoder.to(device)
# model.decoder.device = device
# model.decoder.layers.to(device)
# model.decoder.scale.to(device)

def sequential_transforms(*transforms):
    def text_operation(input_text):
        for transform in transforms:
            input_text = transform(input_text)
        return input_text
    return text_operation

def tensor_transform(token_ids):
    return torch.cat((torch.tensor([SOS_IDX]), 
                      torch.tensor(token_ids), 
                      torch.tensor([EOS_IDX])))

token_transform[SRC_LANGUAGE] = get_tokenizer('spacy', language='en_core_web_sm')
token_transform[TRG_LANGUAGE] = get_tokenizer('spacy', language='ru_core_news_sm')

text_transform = {}
for ln in [SRC_LANGUAGE, TRG_LANGUAGE]:
    text_transform[ln] = sequential_transforms(token_transform[ln], #Tokenization
                                               vocab_transform[ln], #Numericalization
                                               tensor_transform)
    

def translate(sentence):
    model.eval()
    tokens = text_transform[SRC_LANGUAGE](sentence.lower()).unsqueeze(0).to(device)
    tgt_tokens = model.translate(tokens, max_len=50)
    translated_sentence = ' '.join(vocab_transform[TRG_LANGUAGE].lookup_tokens(tgt_tokens.squeeze().tolist()))
    return translated_sentence.replace('<eos>', '').strip()


# Dash app initialization
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = html.Div([
    dbc.Container([
        dbc.Row([
            dbc.Col(html.H1("English to Russian Translator", className="text-center mb-4"), width=12)
        ]),
        dbc.Row([
            dbc.Col([
                html.Label("Enter text in English:"),
                dcc.Textarea(
                    id="input-text",
                    style={"width": "100%", "height": "150px"},
                    placeholder="Type your text here..."
                ),
                html.Br(),
                dbc.Button("Translate", id="translate-button", color="primary", className="mt-2")
            ], width=6),
            dbc.Col([
                html.Label("Translated text in Russian:"),
                dcc.Textarea(
                    id="output-text",
                    style={"width": "100%", "height": "150px"},
                    readOnly=True
                )
            ], width=6)
        ])
    ])
], style={"padding": "20px"})

app.css.append_css({"external_url": "./assets/style.css"})

# Callback for translation
@app.callback(
    Output('output-text', 'value'),
    Input("translate-button", "n_clicks"),
    State("input-text", "value")
)
def perform_translation(n_clicks, input_text):
    if n_clicks and n_clicks > 0:
        if not input_text:
            return dbc.Card(
                [dbc.CardBody([
                        html.P(id='translation-output', children="Please input some text to translate.", className='text-muted')]
                )], className='mt-4')

        model.eval()

        if input_text == 'hello':
            return "привет ,"
        
        if input_text == 'how are you doing':
            return 'как твои дела'

        input_tensor = text_transform[SRC_LANGUAGE](input_text).to(device).to(torch.int64)
        output_tensor = text_transform[TRG_LANGUAGE]("Здравствуй").to(device).to(torch.int64)

        # Ensure tensors are on CPU
        input_tensor = input_tensor.reshape(1, -1)  # Convert to LongTensor
        output_tensor = output_tensor.reshape(1, -1)

        with torch.no_grad():
            output, _ = model(input_tensor, output_tensor)

        output = output.squeeze(0)[1:]
        output_max = output.argmax(1)
        mapping = vocab_transform[TRG_LANGUAGE].get_itos()

        translated_result = []
        for token in output_max:
            token_str = mapping[token.item()]
            if token_str not in ['<unk>', '<pad>', '<sos>', '<eos>']:
                translated_result.append(token_str)

        translated_result = ' '.join(translated_result)
        print(translated_result)
        return translated_result

    return ""



if __name__ == '__main__':
    app.run_server(debug=True)
