'''
 Copyright (c) Alibaba, Inc. and its affiliates.
 @file    chat_format_utils.py
'''
def encode_string_to_tokens(
    message: str, tokenizer, chat_format, system: str = "You are a helpful assistant."
):
    if chat_format == "HUMAN-ASSISTANT":
        raw_text = f"\n\nHuman: {message}\n\nAssistant:"
        context_tokens = tokenizer.tokenize(raw_text)
    elif chat_format == "CHATML":
        im_start, im_end = "<|im_start|>", "<|im_end|>"
        im_start_tokens = tokenizer.tokenize_with_special_tokens(im_start)
        im_end_tokens = tokenizer.tokenize_with_special_tokens(im_end)
        nl_tokens = tokenizer.tokenize("\n")

        system_text = f"system\n{system}"
        system_tokens = (
            im_start_tokens + tokenizer.tokenize(system_text) + im_end_tokens
        )

        raw_text = ""
        context_tokens = []

        context_tokens = (
            system_tokens
            + context_tokens
            + (
                nl_tokens
                + im_start_tokens
                + tokenizer.tokenize(f"user\n{message}")
                + im_end_tokens
                + nl_tokens
                + im_start_tokens
                + tokenizer.tokenize("assistant\n")
            )
        )
        raw_text = (
            f"{im_start}{system_text}{im_end}"
            + raw_text
            + f"\n{im_start}user\n{message}{im_end}\n{im_start}assistant\n"
        )
    raw_text_len = len(raw_text)
    context_length = len(context_tokens)
    return raw_text, context_tokens, raw_text_len, context_length


def _decode_default(tokens, tokenizer, raw_text_len, stop_word="Human:"):
    trim_decode_tokens = tokenizer.detokenize(tokens)[raw_text_len:]
    trim_decode_tokens = trim_decode_tokens.split("<|endoftext|>")[0]
    trim_decode_tokens = trim_decode_tokens.split(stop_word)[0]
    return trim_decode_tokens


def _decode_chatml(
    tokens, stop_words, eod_words, tokenizer, raw_text_len, context_length
):
    eod_token_ids = []
    for eod_word in eod_words:
        eod_token_id = tokenizer.tokenize_with_special_tokens(eod_word)
        assert len(eod_token_id) == 1
        eod_token_ids.append(eod_token_id[0])

    f"Gen length {len(tokens)}"
    eod_token_idx = context_length
    for eod_token_idx in range(context_length, len(tokens)):
        if tokens[eod_token_idx] in eod_token_ids:
            f"Gen {tokenizer.detokenize([tokens[eod_token_idx]])!r}"
            break

    trim_decode_tokens = tokenizer.detokenize(tokens[:eod_token_idx])[raw_text_len:]

    for stop_word in stop_words:
        trim_decode_tokens = trim_decode_tokens.replace(stop_word, "").strip()
    trim_decode_tokens = trim_decode_tokens.strip()

    return trim_decode_tokens


def decode_tokens_to_string(
    tokens, raw_text_len, context_length, tokenizer, chat_format
):
    if chat_format.upper() == "HUMAN-ASSISTANT":
        return _decode_default(tokens, tokenizer, raw_text_len)
    elif chat_format.upper() == "CHATML":
        return _decode_chatml(
            tokens,
            stop_words=[],
            eod_words=["<|endoftext|>", "<|im_end|>"],
            tokenizer=tokenizer,
            raw_text_len=raw_text_len,
            context_length=context_length,
        )
