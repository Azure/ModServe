import pandas as pd
import numpy as np
from datasets import load_dataset
import time
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
TRACES_DIR = REPO_ROOT / "traces"

TRACE_FILE_NAME = sys.argv[1]
SCALE_FACTOR = float(sys.argv[2])
# Optional 3rd arg: window length in minutes carved from the trace start (default 15).
WINDOW_MINUTES = float(sys.argv[3]) if len(sys.argv) > 3 else 15.0
MAX_NUM_IMAGES_PER_PROMPT = 16

# load the timestamped traces
df = pd.read_csv(TRACES_DIR / TRACE_FILE_NAME)
df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'])

start_time = df['TIMESTAMP'].min() # pd.Timestamp('2024-01-01 00:00:00')
mask = (df['TIMESTAMP'] >= start_time) & (df['TIMESTAMP'] <= start_time + pd.Timedelta(minutes=WINDOW_MINUTES))
indices = df[mask].index

df = df.iloc[indices]

if SCALE_FACTOR >= 1:
    print(f"Scaling down the trace by a factor of {SCALE_FACTOR}")

    # support non integers
    num_rows = len(df)
    target_size = int(num_rows / SCALE_FACTOR)
    indices = np.linspace(0, num_rows - 1, target_size, dtype=int)
    df = df.iloc[indices].copy()
else:
    print("Scaling up the intensity of the trace by a factor of", SCALE_FACTOR, 'inverse scale:', 1 / SCALE_FACTOR)
    expanded_df = []
    inverse_scale = 1 / SCALE_FACTOR

    for i in range(len(df)):
        # Calculate the target index in the new dataframe for the start and end of this row
        start_idx = int(np.floor(i * inverse_scale))
        end_idx = int(np.floor((i + 1) * inverse_scale))
        repeats = end_idx - start_idx

        # Add the row the appropriate number of times
        for _ in range(repeats):
            expanded_df.append(df.iloc[i].copy())

    # Convert the list of rows back to a dataframe
    df = pd.DataFrame(expanded_df, columns=df.columns)

# reset the index
df = df.reset_index(drop=True)
pd.set_option('display.max_rows', None)


df['Timestamp'] = pd.to_datetime(df['TIMESTAMP'])
df['Timestamp'] = df['Timestamp'].dt.floor('T')
df['num_requests'] = 1
df_text = df[df['num_images'] == 0]
df_text
df_img = df[df['num_images'] > 0]
df_text = df_text.groupby('Timestamp').agg({'num_images': 'sum', 'n_context_tokens': 'sum', 'num_requests': 'sum'}).reset_index()
df_img = df_img.groupby('Timestamp').agg({'num_images': 'sum', 'n_context_tokens': 'sum', 'num_requests': 'sum'}).reset_index()


# print the average QPS
print('Average text num_requests per min:', df_text['num_requests'].mean())
print('Average image num_requests per min:', df_img['num_requests'].mean())


df['Timestamp'] = pd.to_datetime(df['TIMESTAMP'])
earliest_timestamp = df['Timestamp'].min()
df['Timestamp'] = (df['Timestamp'] - earliest_timestamp).dt.total_seconds()
timestamps = df['Timestamp'].tolist()
for idx, ts in enumerate(timestamps):
    timestamps[idx] = int(ts)

ts_duration = max(timestamps) - min(timestamps)
print('Duration of the trace:', ts_duration, 'seconds', '| max_ts:', max(timestamps), 'min_ts:', min(timestamps))
print('Total requests:', len(df))

df.drop('Timestamp', axis=1)

ds = load_dataset("OpenGVLab/ShareGPT-4o", "image_caption")
df['image_ids'] = None
counter = 0
def get_random_images(num_images):
    global counter
    images = []
    actual_num = min(int(num_images), MAX_NUM_IMAGES_PER_PROMPT)
    for _ in range(actual_num):
        images.append(ds['images'][counter % len(ds['images'])]['image'])
        counter += 1
    return images


def get_text_prompts(num_images):
    row = ds['images'][np.random.randint(0, len(ds['images']))]
    for conversation in row['conversations']:
        src = conversation['from']
        if src != 'human':
            continue
        text_prompt = conversation['value']
        # remove any `<image>` from text prompt
        text_prompt = text_prompt.replace('<image>', '')
        text_prompt = text_prompt.strip()

    return text_prompt

df['image_ids'] = df['num_images'].apply(get_random_images)
df['image_ids'] = df['image_ids'].apply(lambda imgs: " ".join(map(str, imgs)))
df['text_prompt'] = df['num_images'].apply(get_text_prompts)

out_dir = TRACES_DIR / "experiment_traces"
os.makedirs(out_dir, exist_ok=True)
df.to_csv(out_dir / f"{TRACE_FILE_NAME[:TRACE_FILE_NAME.find('.')]}_scale{SCALE_FACTOR}.csv",
          index=False,
          header=True,)

# df2 = pd.read_csv('filename.csv')

