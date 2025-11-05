import os
import pandas as pd
from shutil import copy2


root_dir = "opogona_moth_data"
N = 200
output_folder_name = "top_moths"

output_dir = os.path.join(root_dir, output_folder_name)
os.makedirs(output_dir, exist_ok=True)

records = []
for dirpath, dirnames, filenames in os.walk(root_dir):
    for fname in filenames:
        if fname.lower().endswith(".csv"):
            fpath = os.path.join(dirpath, fname)
            with open(fpath, "r") as fh:
                row_count = sum(1 for _ in fh) - 1
            records.append({"path": fpath, "rows": row_count})

df = pd.DataFrame(records)
df_sorted = df.sort_values("rows", ascending=False).reset_index(drop=True)
top_df = df_sorted.head(N)

for fpath in top_df["path"]:
    copy2(fpath, output_dir)

print(f"Copied {len(top_df)} files into: {output_dir}")
