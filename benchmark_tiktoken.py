import os
import glob
import time
import pandas as pd
import numpy as np
import tiktoken
from tqdm import tqdm
import gzip, shutil

DATASET_PATH = './data/'
OUTPUT_PATH = './output/'
os.makedirs(OUTPUT_PATH, exist_ok=True)

dataset_files = sorted(glob.glob(os.path.join(DATASET_PATH, 'enwik*')))
DATASETS = {os.path.basename(f): f for f in dataset_files}

encoding = tiktoken.get_encoding('cl100k_base')
results = []

for name, path in tqdm(DATASETS.items(), desc="Datasets", unit="file", colour='magenta'):
    step_times = {}

    t0 = time.time()
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        data = f.read()
    step_times['Read'] = time.time() - t0

    t0 = time.time()
    token_ids = encoding.encode(data)
    step_times['Tokenize'] = time.time() - t0

    ids_path = os.path.join(OUTPUT_PATH, f"{name}_tiktoken_ids.npy")
    t0 = time.time()
    np.save(ids_path, np.array(token_ids, dtype=np.int32))
    step_times['SaveIDs'] = time.time() - t0

    t0 = time.time()
    with open(ids_path, 'rb') as f_in, gzip.open(ids_path + '.gz', 'wb', compresslevel=9) as f_out:
        shutil.copyfileobj(f_in, f_out)
    step_times['GzipIDs'] = time.time() - t0
    ids_size = os.path.getsize(ids_path)
    ids_gz_size = os.path.getsize(ids_path + '.gz')

    t0 = time.time()
    uniq_ids = sorted(set(token_ids))
    # Inner tqdm for unique token decoding
    uniq_tokens = []
    for tid in tqdm(uniq_ids, desc=f"[{name}] Decoding unique tokens", unit="tok", leave=False, colour='green'):
        uniq_tokens.append(encoding.decode([tid]))
    uniq_tokens_path = os.path.join(OUTPUT_PATH, f"{name}_uniq_tokens.txt")
    with open(uniq_tokens_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(uniq_tokens))
    step_times['UniqTokens'] = time.time() - t0

    t0 = time.time()
    with open(uniq_tokens_path, 'rb') as f_in, gzip.open(uniq_tokens_path + '.gz', 'wb', compresslevel=9) as f_out:
        shutil.copyfileobj(f_in, f_out)
    step_times['GzipTokens'] = time.time() - t0
    uniq_tokens_size = os.path.getsize(uniq_tokens_path)
    uniq_tokens_gz_size = os.path.getsize(uniq_tokens_path + '.gz')

    results.append({
        'Dataset': name,
        'Raw Size (MB)': round(os.path.getsize(path)/(1024*1024),2),
        'Token Count (M)': round(len(token_ids)/1e6,2),
        'Unique Tokens': len(uniq_ids),
        'IDs Size (MB)': round(ids_size/(1024*1024),2),
        'IDs.gz Size (MB)': round(ids_gz_size/(1024*1024),2),
        'UniqTokens Size (KB)': round(uniq_tokens_size/1024,1),
        'UniqTokens.gz (KB)': round(uniq_tokens_gz_size/1024,1),
        'TimeRead': round(step_times['Read'], 2),
        'TimeTokenize': round(step_times['Tokenize'], 2),
        'TimeSaveIDs': round(step_times['SaveIDs'], 2),
        'TimeGzipIDs': round(step_times['GzipIDs'], 2),
        'TimeUniqTokens': round(step_times['UniqTokens'], 2),
        'TimeGzipTokens': round(step_times['GzipTokens'], 2),
        'TimeTotal': round(sum(step_times.values()), 2)
    })

df = pd.DataFrame(results)
print(df.to_markdown(index=False))
df.to_csv(os.path.join(OUTPUT_PATH, 'tiktoken_benchmark_sizes.csv'), index=False)
print("Benchmark + size report complete. Compressed results in ./output/")

