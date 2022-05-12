import gzip
import json

def text_output(text):
    with gzip.open('jawiki-country.json.gz', mode='rb') as f:
        for line in f:
            obj = json.loads(line)
            if obj['title'] == text:
                return obj['text']