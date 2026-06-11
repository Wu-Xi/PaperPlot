# -*- coding: utf-8 -*-
"""
将 DCF_embeddings 目录中的高维 embedding（例如 64 维）批量降维到 2 维。

脚本会逐个读取 .npy 文件，使用 t-SNE 生成二维坐标，并保存到
DCF_embeddings_tsne2d 目录，输出文件名格式为：
原文件名_tsne2d.npy。
"""

from pathlib import Path
import os
import numpy as np
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize
from tqdm import tqdm

DEFAULT_INPUT_DIR = Path("DCF_embeddings")
DEFAULT_OUTPUT_DIR = Path("DCF_embeddings_tsne2d")
DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
RANDOM_SEED = 2026

def load_embedding(path):
    return np.load(path).astype(np.float32, copy=False)

def tsne_to_2d(embeddings, perplexity=30.0):
    if embeddings.shape[1] == 2:
        return embeddings

    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate='auto',
        random_state=RANDOM_SEED
    )
    return tsne.fit_transform(embeddings).astype(np.float32, copy=False)


for i_file in tqdm(os.listdir(DEFAULT_INPUT_DIR)):
    if i_file.endswith(".npy"):
        print(f"Processing {i_file}...")
        embedding = load_embedding(DEFAULT_INPUT_DIR / i_file)
        embedding_2d = tsne_to_2d(embedding, perplexity=30)
        # embedding_2d = normalize(embedding_2d, axis=0, norm='l2')
        np.save(DEFAULT_OUTPUT_DIR / f"{Path(i_file).stem}_tsne2d.npy", embedding_2d)
