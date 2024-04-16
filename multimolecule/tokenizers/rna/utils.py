def get_vocab_list():
    return VOCAB_LIST


def get_special_tokens_map():
    return SPECIAL_TOKENS_MAP


def get_tokenizer_config():
    config = TOKENIZER_CONFIG
    config.setdefault("added_tokens_decoder", {})
    for i, v in enumerate(SPECIAL_TOKENS_MAP.values()):
        config["added_tokens_decoder"][str(i)] = v
    return config


VOCAB_LIST = [
    "<pad>",
    "<cls>",
    "<eos>",
    "<unk>",
    "<mask>",
    "<null>",
    "A",
    "C",
    "G",
    "U",
    "N",
    "X",
    "V",
    "H",
    "D",
    "B",
    "M",
    "R",
    "W",
    "S",
    "Y",
    "K",
    ".",
    "*",
    "-",
]

SPECIAL_TOKENS_MAP = {
    "pad_token": {
        "content": "<pad>",
        "lstrip": False,
        "normalized": False,
        "rstrip": False,
        "single_word": False,
        "special": True,
    },
    "cls_token": {
        "content": "<cls>",
        "lstrip": False,
        "normalized": False,
        "rstrip": False,
        "single_word": False,
        "special": True,
    },
    "eos_token": {
        "content": "<eos>",
        "lstrip": False,
        "normalized": False,
        "rstrip": False,
        "single_word": False,
        "special": True,
    },
    "unk_token": {
        "content": "<unk>",
        "lstrip": False,
        "normalized": False,
        "rstrip": False,
        "single_word": False,
        "special": True,
    },
    "mask_token": {
        "content": "<mask>",
        "lstrip": False,
        "normalized": False,
        "rstrip": False,
        "single_word": False,
        "special": True,
    },
}

TOKENIZER_CONFIG = {
    "tokenizer_class": "RnaTokenizer",
    "clean_up_tokenization_spaces": True,
}
