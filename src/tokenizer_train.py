

from tokenizers import SentencePieceBPETokenizer

special_tokens = ["<s>", "<pad>", "</s>", "<unk>", "<cls>", "<sep>", "<mask>"]
tokenizer = SentencePieceBPETokenizer()
tokenizer.train_from_iterator(
    text,
    vocab_size=30_000,
    min_frequency=5,
    show_progress=True,
    limit_alphabet=500,
)
tk_tokenizer.save(tokenizer_path)