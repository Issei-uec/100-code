import gzip
import json

def text_output(text):
    with gzip.open('jawiki-country.json.gz', mode='rb') as f:
        for line in f:
            article = json.loads(line)
            if article['title'] == text:
                return article['text']