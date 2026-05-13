from datasets import load_dataset

def load_aime(name="MathArena/aime_2025", split="train", max_examples=None):
    ds = load_dataset(name, split=split)
    if max_examples is not None:
        ds = ds.select(range(min(max_examples, len(ds))))
    return ds
