import os
from pathlib import Path
import pickle
import torch
import scipy.io as sio
from transformers import CLIPTokenizer, CLIPTextModel

@torch.no_grad()
def main():
    repo_root = Path(__file__).resolve().parents[1]
    out_dir = repo_root / "label_embedding"
    out_dir.mkdir(parents=True, exist_ok=True)

    cars_root = Path.home() / "datasets" / "stanford_cars" / "raw"
    devkit = cars_root / "devkit"
    meta = sio.loadmat(devkit / "cars_meta.mat", squeeze_me=True, struct_as_record=False)
    class_names = [str(x).strip() for x in meta["class_names"]]  # 196

    # 1) save label txt
    (out_dir / "stanford_cars_label.txt").write_text("\n".join(class_names), encoding="utf-8")

    # 2) CLIP text embeddings (512-d)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()

    prompts = [f"a photo of a {name}" for name in class_names]
    toks = tokenizer(prompts, padding=True, truncation=True, return_tensors="pt").to(device)
    emb = model(**toks).pooler_output  # [196, 512]
    emb = emb / emb.norm(dim=-1, keepdim=True)

    with open(out_dir / "stanford_cars_le.pickle", "wb") as f:
        pickle.dump(emb.cpu(), f)

    print("Saved:", out_dir / "stanford_cars_le.pickle", "shape=", tuple(emb.shape))

if __name__ == "__main__":
    main()
