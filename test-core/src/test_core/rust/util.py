import os

def dataset_path(name):
    return os.path.join(os.environ["DATASET_DIR"], name)

def cache_dir():
    return os.environ["CACHE_DIR"]
