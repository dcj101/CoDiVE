from huggingface_hub import HfApi

api = HfApi()
names = [
    "jinyoungkim/NExT-GQA",
    "appletea2333/nextgqa",
    "sming256/NeXTGQA",
]

for name in names:
    print("=" * 100)
    print(name)
    try:
        files = api.list_repo_files(name, repo_type="dataset")
        for f in files[:80]:
            print(" ", f)
        print("total files:", len(files))
    except Exception as e:
        print("FAILED:", type(e).__name__, e)