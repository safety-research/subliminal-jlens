#!/usr/bin/env python3
"""Download the public lens files the screen reads into OUTDIR/<repo with '/' -> '__'>/<path> (lens_screen/lenses/ of the
original runs). The original files were downloaded on 2026-10-09; the repositories' revisions at the time this release was
cut were neuronpedia/jacobian-lens e166deb5ad54 and camilablank/workspace-lenses d740106d1e0f (pass --revision to pin). Sizes
range from 0.4 GB (gpt-oss-20b) to 10.6 GB (Llama-3.3-70B); about 45 GB in all.
usage: python planned_lens_strength_screen/fetch_lenses.py OUTDIR [NAME ...]      (default: every lens below)
"""
import argparse
import os

NP, CB = "neuronpedia/jacobian-lens", "camilablank/workspace-lenses"
LENSES = {  # name used in run.sh: (repo, path in repo)
    "llama33_70b__np": (NP, "llama3.3-70b-it/jlens/Salesforce-wikitext/Llama-3.3-70B-Instruct_jacobian_lens.pt"),
    "olmo31_32b__npbase": (NP, "olmo-3-1125-32b/jlens/Salesforce-wikitext/Olmo-3-1125-32B_jacobian_lens.pt"),
    "olmo31_32b__jppbase": (NP, "olmo-3-1125-32b/jpp/Salesforce-wikitext/Olmo-3-1125-32B_jpp_lens.pt"),
    "gemma3_27b__np": (NP, "gemma-3-27b-it/jlens/Salesforce-wikitext/gemma-3-27b-it_jacobian_lens.pt"),
    "gemma3_27b__cbj": (CB, "gemma-3-27b-it/j-lens/lens.pt"),
    "gemma3_27b__cbr": (CB, "gemma-3-27b-it/r-lens/lens.pt"),
    "dsv4__np": (NP, "deepseek-v4-flash/jlens/NeelNanda-pile-10k/deepseek-v4-flash_jacobian_lens_cblank.pt"),
    "dsv4__cbr": (CB, "deepseek-v4-flash/r-lens/lens.pt"),
    "qwen36_27b__np": (NP, "qwen3.6-27b/jlens/Salesforce-wikitext/Qwen3.6-27B_jacobian_lens_n1000.pt"),
    "qwen36_27b__jpp": (NP, "qwen3.6-27b/jpp/Salesforce-wikitext/Qwen3.6-27B_jpp_lens.pt"),
    "qwen35_27b__np": (NP, "qwen3.5-27b/jlens/Salesforce-wikitext/Qwen3.5-27B_jacobian_lens.pt"),
    "gptoss_20b__np": (NP, "gpt-oss-20b/jlens/Salesforce-wikitext/gpt-oss-20b_jacobian_lens.pt"),
}


def local_path(outdir, name):
    repo, path = LENSES[name]
    return os.path.join(outdir, repo.replace("/", "__"), path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("outdir")
    ap.add_argument("names", nargs="*")
    ap.add_argument("--revision", default=None)
    a = ap.parse_args()
    from huggingface_hub import hf_hub_download
    for name in a.names or LENSES:
        repo, path = LENSES[name]
        hf_hub_download(repo, path, revision=a.revision, local_dir=os.path.join(a.outdir, repo.replace("/", "__")))
        print(name, "->", local_path(a.outdir, name), flush=True)
