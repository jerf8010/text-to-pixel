import pytest

from pixelgen.data.tokenizer import TextTokenizer

WORDS = ["a", "red", "circle", "small", "on", "the", "left"]


def make_tok(words=WORDS):
    return TextTokenizer(words)


def test_roundtrip():
    # encode y luego decode devuelve el caption original
    tok = make_tok()
    caption = "a small red circle on the left"
    ids = tok.encode(caption, max_len=len(caption.split()) + 2)
    assert tok.decode(ids) == caption


def test_padding_length_and_value():
    # la salida mide exactamente max_len y lo que sobra es pad_id
    tok = make_tok()
    caption = "a small red circle on the left"
    n_words = len(caption.split())
    max_len = n_words + 2
    ids = tok.encode(caption, max_len=max_len)
    assert len(ids) == max_len
    assert ids[-2:] == [tok.pad_id] * 2
    assert tok.pad_id not in ids[:n_words]


def test_unknown_word_maps_to_unk():
    tok = make_tok()
    ids = tok.encode("a red dragon", max_len=4)
    # "dragon" no está en WORDS
    assert ids[2] == tok.unk_id
    # las palabras conocidas no deben salir como <unk>
    assert ids[0] != tok.unk_id and ids[1] != tok.unk_id


def test_too_long_caption_raises():
    tok = make_tok()
    caption = "a small red circle on the left"
    with pytest.raises(ValueError):
        tok.encode(caption, max_len=len(caption.split()) - 2)


def test_ids_do_not_depend_on_word_order():
    tok = make_tok()
    tok_inv = make_tok(WORDS[::-1])
    assert WORDS != WORDS[::-1]
    assert tok.token_to_id == tok_inv.token_to_id


def test_vocab_size_counts_specials():
    assert make_tok().vocab_size == len(WORDS) + 2