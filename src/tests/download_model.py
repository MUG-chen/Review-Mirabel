import os
from huggingface_hub import HfApi, hf_hub_download
from tqdm import tqdm

# --- 我们需要下载的模型列表 ---
model_ids = [
    "gpt2-xl",
    "roberta-large"
]

# --- 初始化标准的 Hugging Face API 客户端 (不使用镜像) ---
api = HfApi()


def download_model_with_progress(repo_id: str):
    """
    使用tqdm手动创建进度条，逐个下载模型文件。
    """
    try:
        # 1. 获取模型仓库中的所有文件列表
        print(f"[{repo_id}] Fetching file list...")
        model_files = api.list_repo_files(repo_id=repo_id)

        print(f"[{repo_id}] Found {len(model_files)} files to download.")

        # 2. 逐个文件下载
        for filename in tqdm(model_files, desc=f"Overall progress for {repo_id}"):
            # 排除LFS文件夹本身和.gitattributes文件
            if filename.endswith(".gitattributes") or filename.startswith("."):
                continue

            try:
                # 使用 hf_hub_download 下载单个文件
                # 这个函数自带一个基于tqdm的不错的块级下载进度条
                hf_hub_download(
                    repo_id=repo_id,
                    filename=filename,
                    resume_download=True,
                )
            except Exception as e:
                print(f"\n[{repo_id}] Failed to download file {filename}. Error: {e}")
                print("Skipping this file and continuing...")

        print(f"--- Successfully downloaded all available files for {repo_id} ---\n")

    except Exception as e:
        print(f"\n--- Failed to process repo {repo_id}. Error: {e} ---\n")


if __name__ == "__main__":
    # 确保没有残留的环境变量影响
    if 'HF_ENDPOINT' in os.environ:
        del os.environ['HF_ENDPOINT']
        print("Removed HF_ENDPOINT environment variable to ensure direct download.")

    for model_id_to_download in model_ids:
        print(f"--- Starting direct download for model: {model_id_to_download} ---")
        download_model_with_progress(repo_id=model_id_to_download)

    print("All specified models have been processed.")