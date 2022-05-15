import gc
import torch
import numpy as np
import random
from transformers import AutoTokenizer, AutoModelForSequenceClassification


def cleanup():
    gc.collect()
    torch.cuda.empty_cache()


def turn_off_grad(model):
    for param in model.parameters():
        param.requires_grad = False


def turn_on_grad(model):
    for param in model.parameters():
        param.requires_grad = True


def load_model(
        model_name=None,
        model_class=AutoModelForSequenceClassification,
        use_cuda=True
):
    if model_name is None:
        raise ValueError('model_name should be provided')
    model = model_class.from_pretrained(model_name)
    if torch.cuda.is_available() and use_cuda:
        model.cuda()
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    return model, tokenizer


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    np.random.seed(seed)
    random.seed(seed)


def static_vars(**kwargs):
    def decorate(func):
        for k in kwargs:
            setattr(func, k, kwargs[k])
        return func
    return decorate
