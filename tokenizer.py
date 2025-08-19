class LatexTokenizer:
    def __init__(self):
        self.specials = ['<PAD>', '<SOS>', '<EOS>', '<UNK>']
        self.vocab = self.specials.copy()
        self.t2i = {tok: i for i, tok in enumerate(self.vocab)}
        self.i2t = {i: tok for tok, i in self.t2i.items()}

    def build_vocab(self, texts):
        toks = set(tok for txt in texts for tok in txt.split())
        self.vocab = self.specials + sorted(toks)
        self.t2i = {tok: i for i, tok in enumerate(self.vocab)}
        self.i2t = {i: tok for tok, i in self.t2i.items()}

    def encode(self, txt):
        return [self.t2i['<SOS>']] + [self.t2i.get(tok, self.t2i['<UNK>']) for tok in txt.split()] + [self.t2i['<EOS>']]

    def decode(self, ids):
        return ' '.join(self.i2t[i] for i in ids if self.i2t[i] not in self.specials)