import os
import pandas as pd

PROCESSED_ROOT = "../data/processed/CSV"
OUT_FILE = "../data/processed/train_sample.csv"

READ_CHUNKSIZE = 5000
ROWS_PER_CLASS = 20000  # start small; increase later


def main():
    folders = sorted(
        d for d in os.listdir(PROCESSED_ROOT)
        if os.path.isdir(os.path.join(PROCESSED_ROOT, d))
    )

    os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)

    first_write = True
    total_written = 0

    for folder in folders:
        path = os.path.join(PROCESSED_ROOT, folder, "processed.csv")
        if not os.path.exists(path):
            print(f"[skip] missing {path}")
            continue

        need = ROWS_PER_CLASS
        print(f"Sampling {folder}: need {need}")

        for chunk in pd.read_csv(path, chunksize=READ_CHUNKSIZE, low_memory=False):
            if need <= 0:
                break

            take = min(len(chunk), need)
            out_chunk = chunk.iloc[:take].copy()

            out_chunk.to_csv(
                OUT_FILE,
                index=False,
                mode="w" if first_write else "a",
                header=first_write,
            )
            first_write = False

            need -= take
            total_written += take

        print(f"  wrote {ROWS_PER_CLASS - need} rows from {folder}")

    print(f"\nDone. Total rows written: {total_written}")
    print(f"Output: {OUT_FILE}")


if __name__ == "__main__":
    main()
