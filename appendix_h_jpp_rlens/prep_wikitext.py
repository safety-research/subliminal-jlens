"""Dump the official fitting prompt pool (jlens.examples.load_wikitext_prompts(1000), the first 1000 WikiText-103
train records of >= 600 chars, streamed in the dataset's fixed order) to JSON so GPU jobs don't stream the Hub.
The router uses all 1000; each lens fit uses the first 64 (jpp_cli's defaults).
usage (jpp_lens repo root, jpp environment, on a machine with network access): python prep_wikitext.py out.json
"""
import json
import sys

from jlens.examples import load_wikitext_prompts

prompts = load_wikitext_prompts(1000)
json.dump(prompts, open(sys.argv[1], "w"))
print(len(prompts), "prompts; first:", prompts[0][:80].replace("\n", " "))
