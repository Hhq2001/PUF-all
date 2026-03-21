import os
import pandas as pd
import numpy as np
from datetime import datetime
import logging

# ===== Parameters =====
MAX_DATA_POINTS = 50
NUM_COLUMNS = 10
LEFT_COLUMN_VALUES = [25, 80, 200, 500, 1000, 5000, 10000, 50000, 100000, 1000000]

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ===== Create Output Folder =====
def create_output_folder(base_path, name):
    path = os.path.join(base_path, name)
    os.makedirs(path, exist_ok=True)
    return path

# ===== Read CSV =====
def read_csv(file_path):
    try:
        return pd.read_csv(file_path, header=None, dtype=str)
    except Exception as e:
        logging.error(f"Error reading {file_path}: {e}")
        return None

# ===== Process Capacitance Data =====
def process_capacitance(df):
    if df.empty or df.shape[1] < 2:
        logging.warning("Invalid data format.")
        return None

    if len(df.iloc[:, 1]) < MAX_DATA_POINTS:
        logging.warning("Not enough data points.")
        return None

    values = df.iloc[:MAX_DATA_POINTS, 1].astype(float) * 1e11
    reshaped = values.values.reshape(-1, NUM_COLUMNS).round(6)
    return reshaped.T

# ===== Generate Bitstream =====
def generate_bitstream(values, indices):
    reordered = [values[i - 1] for i in indices if i - 1 < len(values)]
    median_value = np.median(reordered)

    bitstream = []
    median_count = reordered.count(median_value)
    current_count = 0

    for v in reordered:
        if v > median_value:
            bitstream.append(1)
        elif v < median_value:
            bitstream.append(0)
        else:
            current_count += 1
            bitstream.append(0 if current_count <= median_count // 2 else 1)

    return bitstream

# ===== Generate Bitstreams for Excel =====
def generate_bitstreams(excel_path, mapping, global_results, output_folder):
    try:
        data = pd.read_excel(excel_path, header=None)
        frequencies = data.iloc[:, 0].dropna().unique()

        folder_name = os.path.basename(os.path.dirname(os.path.dirname(excel_path)))
        indices = mapping.get(folder_name)

        if indices is None:
            logging.warning(f"{folder_name} not found in mapping.")
            return

        results = []

        for freq in frequencies:
            row = data.loc[data.iloc[:, 0] == freq].iloc[0, 1:].dropna().values.astype(float)
            bitstream = generate_bitstream(row, indices)

            results.append(f"Frequency: {freq}, Bitstream: {bitstream}")

            if freq == 500:
                global_results.setdefault(folder_name, {})[os.path.basename(excel_path)] = bitstream

        out_path = os.path.join(output_folder, os.path.basename(excel_path) + "_bitstreams.txt")
        with open(out_path, "w") as f:
            f.write("\n".join(results))

        logging.info(f"Saved: {out_path}")

    except Exception as e:
        logging.error(f"Error generating bitstreams: {e}")

# ===== Save 500Hz Results =====
def save_500hz(global_results, root):
    for folder, data in global_results.items():
        out_dir = os.path.join(root, folder)
        os.makedirs(out_dir, exist_ok=True)

        path = os.path.join(out_dir, "500Hz_Bitstreams.txt")
        with open(path, "w") as f:
            for bs in data.values():
                f.write(''.join(map(str, bs)) + "\n")

# ===== Main Processing =====
def process_directory(root_folder, mapping):
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = create_output_folder(root_folder, f"Capacitance_Result_{now}")

    all_bitstreams_folder = create_output_folder(output_root, "All_Bitstreams")

    means_dict = {}
    global_results = {}

    for dirpath, _, files in os.walk(root_folder):
        for file in files:
            if file.endswith(".csv"):
                path = os.path.join(dirpath, file)
                df = read_csv(path)

                if df is None:
                    continue

                result = process_capacitance(df)
                if result is None:
                    continue

                means = [row.mean() for _, row in pd.DataFrame(result).iterrows()]
                rel_path = os.path.relpath(dirpath, root_folder)
                means_dict.setdefault(rel_path, []).extend(means)

    for rel_path, values in means_dict.items():
        arr = np.array(values)
        rows = (len(arr) + NUM_COLUMNS - 1) // NUM_COLUMNS
        padded = np.pad(arr, (0, rows * NUM_COLUMNS - len(arr)), constant_values=np.nan)
        reshaped = padded.reshape(rows, NUM_COLUMNS)

        df = pd.DataFrame(reshaped).T
        left = pd.DataFrame(LEFT_COLUMN_VALUES)
        final = pd.concat([left, df], axis=1)

        out_dir = create_output_folder(output_root, rel_path)
        out_file = os.path.join(out_dir, f"Result_{rel_path.replace(os.sep, '_')}.xlsx")

        final.to_excel(out_file, index=False, header=False)

        generate_bitstreams(out_file, mapping, global_results, all_bitstreams_folder)

    save_500hz(global_results, output_root)

# ===== Load Mapping =====
def load_mapping(path):
    df = pd.read_excel(path, header=None, dtype=str)
    mapping = {}
    for _, row in df.iterrows():
        mapping[row[0]] = list(map(int, row[1].replace('，', ',').split(',')))
    return mapping

# ===== Entry =====
if __name__ == "__main__":
    mapping = load_mapping("your_mapping.xlsx")
    root_folder = "your_data_folder"
    process_directory(root_folder, mapping)