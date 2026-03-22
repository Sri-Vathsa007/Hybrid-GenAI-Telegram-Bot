from src.bot.utils import extract_keywords, split_text


def test_split_text_returns_multiple_chunks_for_long_input():
    text = " ".join(["token"] * 1200)
    chunks = split_text(text, chunk_size=200, overlap=20)
    assert len(chunks) > 1
    assert all(isinstance(chunk, str) and chunk for chunk in chunks)


def test_extract_keywords_returns_non_empty_keywords():
    caption = "a laptop on a wooden desk beside a notebook and coffee mug"
    keywords = extract_keywords(caption, top_n=3)
    assert len(keywords) == 3
    assert "laptop" in keywords
