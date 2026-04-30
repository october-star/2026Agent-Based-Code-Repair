from datasets import load_dataset

def load_aime():
    ds = load_dataset("MathArena/aime_2025", split="train")
    return ds