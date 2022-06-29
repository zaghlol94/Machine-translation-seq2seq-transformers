import torch
import torch.nn as nn
import pickle
import math
import spacy
from utils import read_lines_from_file, evaluate, bleu
from dataset import get_loader
from config import config
from model import Encoder, Decoder, Seq2Seq

src_test = read_lines_from_file(config["test_config"]["src_test"])
trg_test = read_lines_from_file(config["test_config"]["trg_test"])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

src_tokenizer = spacy.load('de_core_news_sm')
trg_tokenizer = spacy.load('en_core_web_sm')

with open('src_vocab.pkl', 'rb') as inp:
    src_vocab = pickle.load(inp)
with open('trg_vocab.pkl', 'rb') as inp:
    trg_vocab = pickle.load(inp)

test_loader, test_dataset = get_loader(config["test_config"]["src_test"], config["test_config"]["trg_test"], src_vocab,
                                       trg_vocab)

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
test_loss = evaluate(model, test_loader, criterion, device)

print(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |')
print("Belu Score: ",
      bleu(src_test[:100], trg_test[:100], src_vocab, trg_vocab, model, src_tokenizer, trg_tokenizer, device)*100)

