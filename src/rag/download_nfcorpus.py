# src/rag/download_nfcorpus.py
from __future__ import annotations
import os
from pathlib import Path
from beir import util
from beir.datasets.data_loader import GenericDataLoader

def main():
    # 项目根目录 = 当前文件的上上上级目录
    project_root = Path(__file__).resolve().parents[2]  # .../ReviewMirabel
    data_root = project_root / "data"                   # .../ReviewMirabel/data
    data_root.mkdir(parents=True, exist_ok=True)

    url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/nfcorpus.zip"
    print("Downloading NFCorpus to:", str(data_root))
    util.download_and_unzip(url, str(data_root))        # 注意：目标为 data，不是 data/nfcorpus

    dataset_dir = data_root / "nfcorpus"                # 解压后会出现 data/nfcorpus
    print("Dataset dir:", str(dataset_dir))

    corpus, queries, qrels = GenericDataLoader(data_folder=str(dataset_dir)).load(split="test")
    print("Corpus size:", len(corpus))
    print("Queries:", len(queries))
    print("Qrels:", len(qrels))

if __name__ == "__main__":
    main()