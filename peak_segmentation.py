import os
import numpy as np
from obspy import read

def process_mseed_file(file_path):
    st = read(file_path)
    for trace in st:
        from utils import aic_picker
        idx, _ = aic_picker(trace.data)
        start_time = trace.stats.starttime + idx / trace.stats.sampling_rate

        # Κόψιμο από το σημείο έναρξης
        trace.trim(starttime=start_time)

        # Εύρεση peak και διάρκειας
        abs_data = np.abs(trace.data)
        peak_idx = np.argmax(abs_data)
        peak_time = trace.stats.starttime + peak_idx / trace.stats.sampling_rate
        duration = peak_time - start_time
        end_time = peak_time + duration

        trace.trim(endtime=end_time)

        # Νέο όνομα αρχείου
        new_file_path = file_path.replace(".mseed", "_peakseg.mseed")
        trace.write(new_file_path, format="MSEED")
        print(f"✅ Saved: {new_file_path}")


def process_all_mseed_files(base_path):
    for root, _, files in os.walk(base_path):
        for file in files:
            if file.endswith("_demean_detrend_instrumentCorrection.mseed"):
                file_path = os.path.join(root, file)
                try:
                    process_mseed_file(file_path)
                except Exception as e:
                    print(f"⚠️ Error in {file_path}: {e}")


# === Παράδειγμα χρήσης ===
# base_path = "/media/iarv/Samsung/Events/2010"
# process_all_mseed_files(base_path)
