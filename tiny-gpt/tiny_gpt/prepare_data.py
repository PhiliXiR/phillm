"""Download public-domain Alice text, excluding Gutenberg's wrapper."""
import json
from pathlib import Path
from urllib.request import urlopen

URL = "https://www.gutenberg.org/ebooks/11.txt.utf-8"


def main():
    with urlopen(URL, timeout=60) as response:
        text = response.read().decode("utf-8-sig").replace("\r\n", "\n").replace("\r", "\n")
    start = text.find("*** START OF THE PROJECT GUTENBERG EBOOK")
    end = text.find("*** END OF THE PROJECT GUTENBERG EBOOK")
    if start < 0 or end <= start:
        raise RuntimeError("Gutenberg boundary markers changed; inspect source before using it")
    body = text[text.index("\n", start) + 1:end].strip() + "\n"
    root = Path("data")
    root.mkdir(exist_ok=True)
    (root / "alice.txt").write_text(body, encoding="utf-8")
    metadata = {"title": "Alice's Adventures in Wonderland", "author": "Lewis Carroll",
                "source": URL, "catalog": "https://www.gutenberg.org/ebooks/11",
                "license": "Public domain in the USA; underlying 1865 text also public domain in Canada. Check local law elsewhere.",
                "processing": "UTF-8, normalized newlines; Gutenberg header/footer removed; no other cleaning",
                "characters": len(body)}
    (root / "source.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved {len(body):,} characters to data/alice.txt")


if __name__ == "__main__":
    main()
