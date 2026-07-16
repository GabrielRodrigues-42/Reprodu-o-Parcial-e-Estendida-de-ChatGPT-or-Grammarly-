from transformers import AutoTokenizer
from gector import GECToR, predict, load_verb_dict
import torch

_original_load = torch.load
def _cpu_load(*args, **kwargs):
    kwargs.setdefault('map_location', torch.device('cpu'))
    return _original_load(*args, **kwargs)
torch.load = _cpu_load

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = GECToR.from_official_pretrained(
    '/app/data/gector-2024-roberta-large.th',
    special_tokens_fix=1,
    transformer_model='roberta-large',
    vocab_path='/app/data/output_vocabulary',
    max_length=80
).to(device)
tokenizer = AutoTokenizer.from_pretrained('roberta-large', add_prefix_space=True)
encode, decode = load_verb_dict('/app/data/verb-form-vocab.txt')

srcs = ['She go to school yesterday.']
corrected = predict(model, tokenizer, srcs, encode, decode,
                     keep_confidence=0.0, min_error_prob=0.0,
                     n_iteration=5, batch_size=1)
print(corrected)