import spacy
import pickle
import torch
import torch.optim as optim
import torch.nn as nn
import time
import math
from utils import read_lines_from_file, train, evaluate, epoch_time, count_parameters, initialize_weights
from model import Encoder, Decoder, Seq2Seq
from vocabulary import Vocabulary
from dataset import get_loader
from config import config

src_train = read_lines_from_file(config["src_train"])
trg_train = read_lines_from_file(config["trg_train"])
src_valid = read_lines_from_file(config["src_valid"])
trg_valid = read_lines_from_file(config["trg_valid"])

print(f"Number of training examples: {len(src_train)}")
print(f"Number of validation examples: {len(src_valid)}")

src_tokenizer = spacy.load('de_core_news_sm')
trg_tokenizer = spacy.load('en_core_web_sm')

src_vocab = Vocabulary(2, src_tokenizer)
src_vocab.build_vocabulary(src_train)

trg_vocab = Vocabulary(2, trg_tokenizer)
trg_vocab.build_vocabulary(trg_train)

with open('src_vocab.pkl', 'wb') as file:
    pickle.dump(src_vocab, file, pickle.HIGHEST_PROTOCOL)

with open('trg_vocab.pkl', 'wb') as file:
    pickle.dump(trg_vocab, file, pickle.HIGHEST_PROTOCOL)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

INPUT_DIM = len(src_vocab.stoi)
OUTPUT_DIM = len(trg_vocab.stoi)
SRC_PAD_IDX = src_vocab.stoi["<pad>"]
TRG_PAD_IDX = trg_vocab.stoi["<pad>"]

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
print(f'The model has {count_parameters(model):,} trainable parameters')

model.apply(initialize_weights)

LEARNING_RATE = config["learning_rate"]

optimizer = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE)
criterion = nn.CrossEntropyLoss(ignore_index = TRG_PAD_IDX)

CLIP = 1

best_valid_loss = float('inf')

for epoch in range(config["N_EPOCHS"]):

    start_time = time.time()
    train_iterator, train_dataset = get_loader(config["src_train"], config["trg_train"], src_vocab, trg_vocab)
    valid_iterator, val_dataset = get_loader(config["src_train"], config["trg_train"], src_vocab, trg_vocab)

    train_loss = train(model, train_iterator, optimizer, criterion, CLIP, device)
    valid_loss = evaluate(model, valid_iterator, criterion, device)

    end_time = time.time()

    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'model.pt')

    print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')
