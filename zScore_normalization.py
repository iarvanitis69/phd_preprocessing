import os
import numpy as np
from obspy import read

def z_score_normalize_mseed(input_path):
    stream = read(input_path)
    data = stream[0].data

    mean = np.mean(data)
    std = np.std(data)
    normalized_data = (data - mean) / std

    stream[0].data = normalized_data

    base, ext = os.path.splitext(input_path)
    output_path = f"{base}_zscore{ext}"

    stream.write(output_path, format='MSEED')
    return output_path

def zscore_normalize_all_files():
    # Αναζήτηση όλων των αρχείων που τελειώνουν σε L/Q/T  και .mseed
    valid_suffixes = ("_L.mseed", "_Q.mseed", "_T.mseed")

    from main import BASE_DIR
    for root, _, files in os.walk(BASE_DIR):
        for filename in files:
            if filename.endswith(valid_suffixes):
                full_path = os.path.join(root, filename)
                try:
                    output = z_score_normalize_mseed(full_path)
                    print(f"✅ Normalized: {output}")
                except Exception as e:
                    print(f"❌ Failed: {full_path} — {e}")

# Παράδειγμα χρήσης:
# zscore_normalize_all_files('/path/to/base/dir')
