from datasets import load_dataset, get_dataset_split_names

names = [
    "appletea2333/nextgqa",
    "sming256/NeXTGQA",
]

for name in names:
    print("=" * 100)
    print("testing:", name)
    try:
        splits = get_dataset_split_names(name)
        print("splits:", splits)

        for split in splits:
            print("-" * 50)
            print("split:", split)
            ds = load_dataset(name, split=split)
            print("num rows:", len(ds))
            print("features:", ds.features)
            print("first row:", ds[0])
    except Exception as e:
        print("FAILED:", type(e).__name__, e)