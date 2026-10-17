"""上传 v2 bundle（不含凭据）到 ModelScope 私有仓库，并尝试删除旧 bundle。"""
from modelscope.hub.api import HubApi

TOKEN = open("/home/czx/RLPD/deployment/.secrets.env") \
    .read().split("MS_TOKEN=")[1].split("\n")[0].strip()

api = HubApi()
api.login(TOKEN)
repo_id = "CCLLUU/rlpd-repro"

# 尝试删除含凭据的旧 bundle
try:
    api.delete_files(repo_id=repo_id, file_paths=["rlpd-rocm-bundle.tar.gz"],
                     commit_message="remove secret-bearing bundle")
    print("old bundle deleted")
except Exception as e:
    print("delete old failed (可网页删):", str(e)[:100])

api.upload_file(
    repo_id=repo_id,
    path_or_fileobj="/home/czx/RLPD/rlpd-rocm-bundle-v2.tar.gz",
    path_in_repo="rlpd-rocm-bundle-v2.tar.gz",
    commit_message="v2 bundle: py3.11 stack + lessons learned, no secrets",
)
print("uploaded:", repo_id, "/rlpd-rocm-bundle-v2.tar.gz")
