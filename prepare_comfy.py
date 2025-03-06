import json
import subprocess
import uuid
from pathlib import Path
from typing import Dict

import modal

image = (  # build up a Modal Image to run ComfyUI, step by step
    modal.Image.debian_slim(  # start from basic Linux with Python
        python_version="3.11"
    )
    .apt_install("git")  # install git to clone ComfyUI
)

# install bcat-civitai
image = (
    # get deps for rust
    image.apt_install("curl", "build-essential", "libssl-dev", "pkg-config")
    .run_commands("curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y")
    .run_commands("git clone https://github.com/burritocatai/bcat-civitai.git")
    .run_commands(". \"$HOME/.cargo/env\" && cd bcat-civitai && cargo build --release")
)

vol = modal.Volume.from_name("model-cache", create_if_missing=True)

image = (
    # install huggingface_hub with hf_transfer support to speed up downloads
    image.pip_install("huggingface_hub[hf_transfer]==0.26.2")
)

image = (
    # install huggingface_hub with hf_transfer support to speed up downloads
    image.pip_install("huggingface_hub[hf_transfer]==0.26.2")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

image = image.add_local_file(
    Path(__file__).parent / "models.txt", "/root/models.txt"
)

app = modal.App(name="prepare-comfyui", image=image)


@app.function(    
    volumes={"/models": vol},  # mounts our cached models
    secrets=[modal.Secret.from_name("civitai-api"), modal.Secret.from_name("huggingface-secret")])
def download_all_models():
    """
    Download models from Civitai or Hugging Face based on URNs listed in the provided file.

    Args:
        models_file (str): Path to a file containing model URNs, one per line.
    """
    with open('/root/models.txt', 'r') as f:
        for line in f:
            urn = line.strip()
            if not urn or urn.startswith("#"):
                continue

            print(f"Downloading {urn}")
            if "civitai" in urn:
                subprocess.run(
                    f"/bcat-civitai/target/release/bcat-civitai --urn {urn} --base-dir /models --comfyui",
                    shell=True,
                    check=True,
                )

            elif ":hf:" in urn:
               split_urn = urn.split(":")
               comfy_dir = split_urn[3]
               filename = split_urn[5].split("@")[1]
               repo_id = split_urn[5].split("@")[0]
               subprocess.run("pip install huggingface_hub[hf_transfer]==0.26.2", shell=True)
               subprocess.run(f"huggingface-cli download {repo_id} {filename} --local-dir /models/{comfy_dir} --local-dir-use-symlinks False --resume-download", shell=True)
            else:
                print(f"Unknown URN format: {urn}")



@app.local_entrypoint()
def main():
    download_all_models.remote()

