"""Download pretrained weights into cache/huggingface on the LOGIN node (compute jobs run with HF_HUB_OFFLINE=1).
timm names are resolved to their HF hub repo; 'hf_hub:<repo>' names (Zoobot) are fetched directly.
Usage: HF_HOME=cache/huggingface python scripts/fetch_weights.py NAME [NAME ...]
"""
import sys

import timm
from huggingface_hub import snapshot_download

for name in sys.argv[1:]:
    try:
        if name.startswith("hf_hub:"):
            repo = name[len("hf_hub:"):]
        else:
            cfg = timm.models.get_pretrained_cfg(name)
            repo = cfg.hf_hub_id
            if not repo:
                print("NO_HF_REPO", name, cfg.url, flush=True)
                continue
        from huggingface_hub import list_repo_files
        files = list_repo_files(repo)
        wpat = ["*.safetensors"] if any(f.endswith(".safetensors") for f in files) else ["*.bin"]
        snapshot_download(repo, allow_patterns=["*.json", "*.md"] + wpat)
        print("OK", name, repo, flush=True)
    except Exception as e:  # noqa: BLE001
        print("FAIL", name, repr(e)[:200], flush=True)
