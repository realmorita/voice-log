from voice_log.summarize import format_surrogate_codepoints, replace_surrogate_codepoints


def test_replace_surrogate_codepoints_replaces_and_reports_positions() -> None:
    raw_text = "A\ud800B\udfffC"

    sanitized_text, occurrences = replace_surrogate_codepoints(raw_text)

    assert sanitized_text == "A\uFFFDB\uFFFDC"
    assert occurrences == [(1, 0xD800), (3, 0xDFFF)]


def test_replace_surrogate_codepoints_returns_original_when_clean() -> None:
    raw_text = "これは安全なテキストです"

    sanitized_text, occurrences = replace_surrogate_codepoints(raw_text)

    assert sanitized_text == raw_text
    assert occurrences == []


def test_format_surrogate_codepoints_limits_output() -> None:
    occurrences = [(0, 0xD800), (10, 0xD801), (20, 0xD802)]

    formatted = format_surrogate_codepoints(occurrences, max_items=2)

    assert formatted == "count=3, positions=0:D800, 10:D801, ... +1"
