import torch

from peft import PeftModel, PeftConfig
from transformers import AutoConfig, AutoModel, AutoTokenizer

def get_embedding_dim(model_name_or_path):
    config = AutoConfig.from_pretrained(model_name_or_path)

    embedding_dim = None
    for attr in ['hidden_size', 'd_model', 'dim', 'embed_dim']:
        if hasattr(config, attr):
            embedding_dim = getattr(config, attr)
            return embedding_dim

    return 768

def load_repllama_model(model_name : str, htf_token : str) -> PeftModel:
    config = PeftConfig.from_pretrained(model_name)
    base_model = AutoModel.from_pretrained(config.base_model_name_or_path, token=htf_token, device_map='auto')
    model = PeftModel.from_pretrained(base_model, model_name)
    model = model.merge_and_unload()
    model.eval()
    model = torch.compile(model)
    return model

def load_repllama_tokenizer(model_name : str, hf_token : str) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
    tokenizer.pad_token_id = tokenizer.unk_token_id
    tokenizer.pad_token = tokenizer.unk_token
    tokenizer.padding_side = 'right'
    return tokenizer

def load_model_and_tokenizer(model_name : str, hf_token : str = None) -> tuple:
    if model_name == 'castorini/repllama-v1-7b-lora-passage':
        return load_repllama_model(model_name, hf_token), load_repllama_tokenizer('meta-llama/Llama-2-7b-hf', hf_token)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = AutoModel.from_pretrained(model_name)
    model = model.to(device)
    model = torch.compile(model)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    return model, tokenizer
