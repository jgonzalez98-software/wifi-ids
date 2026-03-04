import os
import pandas as pd

RAW_ROOT = "../data/raw/CSV"
OUT_ROOT = "../data/processed/CSV"

READ_CHUNKSIZE = 5000   # smaller to avoid crashes
CLASS_FOLDER = "13.Website_spoofing"


def process_one_class(class_folder: str):
    in_dir = os.path.join(RAW_ROOT, class_folder)
    if not os.path.isdir(in_dir):
        raise RuntimeError(f"Input folder not found: {in_dir}")

    out_dir = os.path.join(OUT_ROOT, class_folder)
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, "processed.csv")

    csv_files = sorted([f for f in os.listdir(in_dir) if f.lower().endswith(".csv")])
    if not csv_files:
        raise RuntimeError(f"No CSV files found in: {in_dir}")

    # attack_id from "1.Deauth" -> 1, else None
    attack_id = None
    if "." in class_folder:
        left = class_folder.split(".", 1)[0]
        if left.isdigit():
            attack_id = int(left)

    print(f"Processing {class_folder}")
    print(f"Input:  {in_dir}")
    print(f"Output: {out_csv}")
    print(f"Files:  {len(csv_files)}")

    first_write = True
    total_rows = 0

    for i, fname in enumerate(csv_files, start=1):
        path = os.path.join(in_dir, fname)
        print(f"[{i}/{len(csv_files)}] {fname}")

        for chunk in pd.read_csv(
            path,
            chunksize=READ_CHUNKSIZE,
            on_bad_lines="skip",
            low_memory=False,
        ):
            # De-fragment (what pandas warning suggests)
            chunk = chunk.copy()

            # Labels so you never mix them up
            chunk["attack_type"] = class_folder
            chunk["attack_id"] = attack_id

            # Write immediately (no concat)
            chunk.to_csv(
                out_csv,
                index=False,
                mode="w" if first_write else "a",
                header=first_write,
            )

            first_write = False
            total_rows += len(chunk)

    print(f"Done. Wrote {total_rows} rows -> {out_csv}")


if __name__ == "__main__":
    process_one_class(CLASS_FOLDER)
