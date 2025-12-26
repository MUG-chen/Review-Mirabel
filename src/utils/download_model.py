# download_model.py
from modelscope import snapshot_download

# 1. 指定模型 ID (魔搭社区的 ID)
model_id = 'AI-ModelScope/roberta-large'

# 'LLM-Research/Meta-Llama-3.1-8B-Instruct'
# BAAI/bge-m3
# BAAI/bge-reranker-large
# openai-community/gpt2
# AI-ModelScope/roberta-large

# 2. 指定下载目录 (建议放在数据盘，如 autodl-tmp)
# 下载后，模型会自动存在 /root/autodl-tmp/pycharm_Mirabel/models 里面
cache_dir = '/root/autodl-tmp/pycharm_Mirabel/models'

print(f"正在从魔搭社区下载 {model_id} 到 {cache_dir} ...")

# 3. 开始下载
model_dir = snapshot_download(model_id, cache_dir=cache_dir)

print(f"\n下载完成！")
print(f"模型实际保存路径是: {model_dir}")
