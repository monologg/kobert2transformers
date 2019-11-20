import sentencepiece as spm
from kobert.utils import get_tokenizer
from transformers import BertTokenizer


class KoBERTTokenizer(object):
    """
    https://github.com/google/sentencepiece/blob/master/python/README.md
    """

    def __init__(self, cache_dir=None):
        self.sp = spm.SentencePieceProcessor()
        if cache_dir:
            self.sp.Load(cache_dir)
        else:
            self.sp.Load(get_tokenizer())

    def tokenize(self, text):
        return self.sp.EncodeAsPieces(text)

    def decode(self, token_ids):
        return self.sp.DecodeIds(token_ids)

    def encode(self, text):
        return self.sp.EncodeAsIds(text)

    def convert_tokens_to_ids(self, tokens):
        ids = []
        for token in tokens:
            ids.append(self.sp.PieceToId(token))
        return ids


if __name__ == "__main__":
    tokenizer = KoBERTTokenizer()

    text = "한국어 모델이당~~ kor"
    a = tokenizer.tokenize(text)
    print(a)
    print(tokenizer.convert_tokens_to_ids(a))
    a = tokenizer.encode(text)
    print(a)
