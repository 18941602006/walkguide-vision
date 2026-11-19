"""Fetch public video models from their original publishers."""

from pathlib import Path

from huggingface_hub import snapshot_download


ROOT = Path(__file__).resolve().parent
MODELS = {
    "model-outdoor": "nvidia/segformer-b2-finetuned-cityscapes-1024-1024",
    "model-indoor": "nvidia/segformer-b2-finetuned-ade-512-512",
}


def main():
    for folder, repository in MODELS.items():
        print(f"Downloading {repository}", flush=True)
        snapshot_download(
            repo_id=repository,
            local_dir=ROOT / folder,
            allow_patterns=[
                "config.json", "preprocessor_config.json", "pytorch_model.bin",
                "README.md", "LICENSE*",
            ],
        )
    print("Models ready. Start with: python web_app.py")


if __name__ == "__main__":
    main()
