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
    .pip_install("fastapi[standard]==0.115.4")  # install web dependencies
    .pip_install("comfy-cli==1.3.5")  # install comfy-cli
    .run_commands(  # use comfy-cli to install ComfyUI and its dependencies
        "comfy --skip-prompt install --nvidia --version 0.3.23"
    )
)

# install bcat-civitai
image = (
    # get deps for rust
    image.apt_install("curl", "build-essential", "libssl-dev", "pkg-config")
    .run_commands("curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y")
    .run_commands("git clone https://github.com/burritocatai/bcat-civitai.git")
    .run_commands(". \"$HOME/.cargo/env\" && cd bcat-civitai && cargo build --release")
)

image = (
    image.run_commands(  # download a custom node
        "comfy node install was-node-suite-comfyui@1.0.2"
    )
    # Add .run_commands(...) calls for any other custom nodes you want to download
)

def hf_download():
    from huggingface_hub import hf_hub_download

    flux_model = hf_hub_download(
        repo_id="Comfy-Org/flux1-schnell",
        filename="flux1-schnell-fp8.safetensors",
        cache_dir="/cache",
    )

    # symlink the model to the right ComfyUI directory
    subprocess.run(
        f"ln -s {flux_model} /root/comfy/ComfyUI/models/checkpoints/flux1-schnell-fp8.safetensors",
        shell=True,
        check=True,
    )


vol = modal.Volume.from_name("hf-hub-cache", create_if_missing=True)

image = (
    # install huggingface_hub with hf_transfer support to speed up downloads
    image.pip_install("huggingface_hub[hf_transfer]==0.26.2")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .run_function(
        hf_download,
        # persist the HF cache to a Modal Volume so future runs don't re-download models
        volumes={"/cache": vol},
    )
)

image = image.add_local_file(
    Path(__file__).parent / "workflow_api.json", "/root/workflow_api.json"
)

app = modal.App(name="example-comfyui", image=image)


@app.function(
    allow_concurrent_inputs=10,  # required for UI startup process which runs several API calls concurrently
    max_containers=1,  # limit interactive session to 1 container
    gpu="L40S",  # good starter GPU for inference
    volumes={"/cache": vol},  # mounts our cached models
    secrets=[modal.Secret.from_name("civitai-api")]
)
@modal.web_server(8000, startup_timeout=60)
def ui():
    bcat_command = "/bcat-civitai/target/release/bcat-civitai "
    bcat_command += "--urn urn:air:flux1:checkpoint:civitai:618692@691639 "
    bcat_command += "--base-dir /root/comfy/ComfyUI/models --comfyui"

    subprocess.Popen(bcat_command, shell=True)
    subprocess.Popen("comfy launch -- --listen 0.0.0.0 --port 8000", shell=True)

