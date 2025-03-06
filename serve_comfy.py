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


if Path("./nodes.txt").exists():
    with open('./nodes.txt', 'r') as f:
        for line in f:
            node = line.strip()
            if not node or node.startswith("#"):
                continue
            image = (image.run_commands(  
                # download a custom node from nodes.txt
                f"comfy node install {node}"
            ))




vol = modal.Volume.from_name("model-cache", create_if_missing=True)

# image = (
#     image.run_commands(
#             "find /models -type f -exec sh -c 'ln -sf \"{}\" \"/root/comfy/ComfyUI/models/$(realpath --relative-to=/models \"{}\")\"' \;",
#     )
# )



image = image.add_local_file(
    Path(__file__).parent / "workflow_api.json", "/root/workflow_api.json"
)

app = modal.App(name="example-comfyui", image=image)

@app.function(
    allow_concurrent_inputs=10,  # required for UI startup process which runs several API calls concurrently
    max_containers=1,  # limit interactive session to 1 container
    gpu="L40S",  # good starter GPU for inference
    volumes={"/models": vol},  # mounts our cached models
    secrets=[modal.Secret.from_name("civitai-api")]
)
@modal.web_server(8000, startup_timeout=60)
def ui():
    import os

    source_dir = '/models'
    target_dir = '/root/comfy/ComfyUI/models'

    # Make sure target directory exists
    os.makedirs(target_dir, exist_ok=True)

    # Walk through source directory
    for root, dirs, files in os.walk(source_dir):
        # Get the relative path from source_dir
        rel_path = os.path.relpath(root, source_dir)
        
        # Create the corresponding target directory if not at the root level
        if rel_path != '.':
            target_subdir = os.path.join(target_dir, rel_path)
            os.makedirs(target_subdir, exist_ok=True)
        else:
            target_subdir = target_dir
        
        # Create symlinks for all files in current directory
        for file in files:
            source_file = os.path.join(root, file)
            target_file = os.path.join(target_subdir, file)
            
            # Remove existing symlink or file if it exists
            if os.path.exists(target_file) or os.path.islink(target_file):
                if os.path.islink(target_file):
                    os.unlink(target_file)
                else:
                    print(f"Warning: {target_file} exists and is not a symlink. Skipping.")
                    continue
            
            # Create the symlink
            os.symlink(source_file, target_file)
            print(f"Created symlink: {target_file} -> {source_file}")

        print("Symlinking complete!")

    subprocess.Popen("comfy launch -- --listen 0.0.0.0 --port 8000", shell=True)

