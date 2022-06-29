import torch
import torch.nn as nn
import pickle
import spacy
import argparse
from utils import translate_sentence, display_attention
from config import config
from model import Encoder, Decoder, Seq2Seq

parser = argparse.ArgumentParser(description="translate string from german to english")
parser.add_argument("-s", "--sentence", type=str, required=True, help="sentence that you want to translate")
args = parser.parse_args()

print(args.sentence)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

src_tokenizer = spacy.load('de_core_news_sm')
trg_tokenizer = spacy.load('en_core_web_sm')

with open('src_vocab.pkl', 'rb') as inp:
    src_vocab = pickle.load(inp)
with open('trg_vocab.pkl', 'rb') as inp:
    trg_vocab = pickle.load(inp)

INPUT_DIM = len(src_vocab.stoi)
OUTPUT_DIM = len(trg_vocab.stoi)
SRC_PAD_IDX = src_vocab.stoi["<pad>"]
TRG_PAD_IDX = trg_vocab.stoi["<pad>"]
criterion = nn.CrossEntropyLoss(ignore_index = TRG_PAD_IDX)

enc = Encoder(INPUT_DIM,
              config["HID_DIM"],
              config["ENC_LAYERS"],
              config["ENC_HEADS"],
              config["ENC_PF_DIM"],
              config["ENC_DROPOUT"],
              device)

dec = Decoder(OUTPUT_DIM,
              config["HID_DIM"],
              config["DEC_LAYERS"],
              config["DEC_HEADS"],
              config["DEC_PF_DIM"],
              config["DEC_DROPOUT"],
              device)

model = Seq2Seq(enc, dec, SRC_PAD_IDX, TRG_PAD_IDX, device).to(device)
model.load_state_dict(torch.load(config["test_config"]["model_path"]))

translation, attention = translate_sentence(
        sentence=args.sentence, src_vocab=src_vocab, trg_vocab=trg_vocab, src_tokenizer=src_tokenizer, model=model, device=device
    )
print(" ".join(translation[:-1]))
