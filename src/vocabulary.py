class Vocabulary:
    def __init__(self, freq_threshold, tokenizer):
        self.itos = {0: "<unk>", 1: "<pad>", 2: "<sos>", 3: "<eos>"}
        self.stoi = {"<unk>": 0, "<pad>": 1, "<sos>": 2, "<eos>": 3}
        self.freq_threshold = freq_threshold
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.itos)

    def tokenize(self, text):
        return [tok.text.lower() for tok in self.tokenizer.tokenizer(text)]

    def build_vocabulary(self, sentence_list):
        frequencies = {}
        idx = 4

        for sentence in sentence_list:
            for word in self.tokenize(sentence):
                if word not in frequencies:
                    frequencies[word] = 1

                else:
                    frequencies[word] += 1

                if frequencies[word] == self.freq_threshold:
                    self.stoi[word] = idx
                    self.itos[idx] = word
                    idx += 1

    def numericalize(self, text):
        tokenized_text = self.tokenize(text)

        return [
            self.stoi[token] if token in self.stoi else self.stoi["<unk>"]
            for token in tokenized_text
        ]
