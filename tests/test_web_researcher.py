import pytest
from research.web_researcher import extract_research_query, parse_user_source_override, _extract_text_from_html

def test_extract_text_from_html_cleans_tags():
    html_sample = """
    <html>
        <head><title>Test</title></head>
        <body>
            <nav>Menu item 1</nav>
            <h1>Artificial Intelligence</h1>
            <p>Artificial Intelligence (AI) is intelligence demonstrated by machines.</p>
            <script>console.log("ignore me");</script>
        </body>
    </html>
    """
    clean_text = _extract_text_from_html(html_sample)
    assert "Artificial Intelligence" in clean_text
    assert "intelligence demonstrated by machines" in clean_text
    assert "ignore me" not in clean_text
    assert "Menu item 1" not in clean_text

def test_extract_text_from_html_table_parsing():
    html_table = """
    <table>
        <tr><th>Language</th><th>Typing</th></tr>
        <tr><td>Python</td><td>Dynamic</td></tr>
        <tr><td>Rust</td><td>Static</td></tr>
    </table>
    """
    extracted = _extract_text_from_html(html_table)
    assert "[TABLE]" in extracted
    assert "Language | Typing" in extracted
    assert "Python | Dynamic" in extracted
