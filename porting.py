import os
import json

import torch
from kobert.pytorch_kobert import get_pytorch_kobert_model, bert_config

MODEL_DIR = 'model'
ORIGINAL_DIR = 'original'

if not os.path.exists(MODEL_DIR):
    os.mkdir(MODEL_DIR)


# Save model
model, _ = get_pytorch_kobert_model()

torch.save(model.state_dict(), os.path.join(MODEL_DIR, 'pytorch_model.bin'))


# Save vocab
vocab = json.load(open(os.path.join(ORIGINAL_DIR, 'kobertvocab.json')))

with open(os.path.join(MODEL_DIR, 'vocab.txt'), 'w', encoding='utf-8') as f:
    for token in vocab["idx_to_token"]:
        f.write(token+"\n")


# Save config
with open(os.path.join(MODEL_DIR, "config.json"), 'w', encoding='utf-8') as f:
    json.dump(bert_config, f, indent=4)
