import torch
import random
from utils import read_lines_from_file
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence  # pad batch


class TranslatePairDataset(Dataset):
    def __init__(self, src_file, trg_file, src_vocab, trg_vocab):
        self.src = read_lines_from_file(src_file)
        self.trg = read_lines_from_file(trg_file)

        self.src_vocab = src_vocab
        self.trg_vocab = trg_vocab

    def __len__(self):
        return len(self.src)

    def __getitem__(self, index):
        src = self.src[index]
        trg = self.trg[index]

        numericalized_src = [self.src_vocab.stoi["<sos>"]]
        numericalized_src += self.src_vocab.numericalize(src)
        numericalized_src.append(self.src_vocab.stoi["<eos>"])

        numericalized_trg = [self.trg_vocab.stoi["<sos>"]]
        numericalized_trg += self.trg_vocab.numericalize(trg)
        numericalized_trg.append(self.trg_vocab.stoi["<eos>"])

        return torch.tensor(numericalized_src), torch.tensor(numericalized_trg)


class MyCollate:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        src = [item[0]for item in batch]
        src = pad_sequence(src, batch_first=True, padding_value=self.pad_idx)
        trg = [item[1] for item in batch]
        trg = pad_sequence(trg, batch_first=True, padding_value=self.pad_idx)

        return src, trg


def get_loader(
        src_file,
        trg_file,
        src_vocab,
        trg_vocab,
        batch_size=64,
        num_workers=8,
        shuffle=True,
        pin_memory=True,
):
    dataset = TranslatePairDataset(src_file, trg_file, src_vocab, trg_vocab)

    def batch_sampler():
        indices = [(i, len(s[0])) for i, s in enumerate(dataset)]
        random.shuffle(indices)
        pooled_indices = []
        # create pool of indices with similar lengths
        for i in range(0, len(indices), batch_size * 100):
            pooled_indices.extend(sorted(indices[i:i + batch_size * 100], key=lambda x: x[1]))

        pooled_indices = [x[0] for x in pooled_indices]

        # yield indices for current batch
        for i in range(0, len(pooled_indices), batch_size):
            yield pooled_indices[i:i + batch_size]

    pad_idx = src_vocab.stoi["<pad>"]
    loader = DataLoader(
        dataset=dataset,
        num_workers=num_workers,
        pin_memory=pin_memory,
        batch_sampler=batch_sampler(),
        collate_fn=MyCollate(pad_idx=pad_idx),
    )

    return loader, dataset
