from src.preprocessor import clean_text


def test_clean_text_lowercase():
    assert clean_text("HELLO WORLD") == "hello world"


def test_clean_text_removes_url():
    result = clean_text("check this https://example.com now")
    assert "https" not in result
    assert "example" not in result


def test_clean_text_removes_mentions():
    result = clean_text("@user hello there")
    assert "@user" not in result
    assert result == "hello there"


def test_clean_text_removes_rt():
    result = clean_text("RT this is a tweet")
    assert result == "this is a tweet"


def test_clean_text_collapses_spaces():
    result = clean_text("hello     world")
    assert result == "hello world"