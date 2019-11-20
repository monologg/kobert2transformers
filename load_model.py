import torch
from transformers import BertModel, BertConfig

bert_config = BertConfig.from_pretrained("model")
model = BertModel.from_pretrained('model', config=bert_config)


input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])
outputs = model(input_ids, input_mask, token_type_ids)
print(outputs[0])
print(outputs[1].shape)