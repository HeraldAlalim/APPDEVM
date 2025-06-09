import os
import json
import numpy as np
from sklearn.model_selection import train_test_split

DATA_DIR = 'data'
SEQUENCE_LENGTH = 30

def load_sequences(data_dir):
    X = []
    y = []

    labels = sorted({f.split('_')[0] for f in os.listdir(data_dir) if f.endswith('.txt')})
    label_map = {label: idx for idx, label in enumerate(labels)}
    print(f"[INFO] Found labels: {label_map}")

    for file_name in os.listdir(data_dir):
        if not file_name.endswith('.txt'):
            continue

        label = file_name.split('_')[0]
        label_idx = label_map[label]

        file_path = os.path.join(data_dir, file_name)
        with open(file_path, 'r') as f:
            lines = f.readlines()
            sequence = [list(map(float, line.strip().split())) for line in lines]

            if len(sequence) < SEQUENCE_LENGTH:
                print(f"[WARNING] Skipping {file_name}: not enough frames ({len(sequence)})")
                continue

            sequence = sequence[:SEQUENCE_LENGTH]
            X.append(sequence)
            y.append(label_idx)

    with open("label_map.json", "w") as f:
        json.dump(label_map, f)

    return np.array(X), np.array(y), label_map

if __name__ == '__main__':
    X, y, label_map = load_sequences(DATA_DIR)

    if len(X) == 0:
        print("[ERROR] No valid sequences found. Please check your .txt files.")
        exit()

    np.save('X_full.npy', X)
    np.save('y_full.npy', y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    np.save('X_train.npy', X_train)
    np.save('y_train.npy', y_train)
    np.save('X_test.npy', X_test)
    np.save('y_test.npy', y_test)

    print(f"[DONE] Saved datasets: X_train.npy, y_train.npy, X_test.npy, y_test.npy")
    print(f"[INFO] Label mapping saved to label_map.json: {label_map}")
