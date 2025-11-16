import os
import numpy as np
from obspy import read


def z_score_normalize_mseed(input_path):
    """
    Apply Z-score normalization to one .mseed file
    and save output as *_zscore.mseed
    """
    stream = read(input_path)
    data = stream[0].data.astype(np.float64)

    mean = np.mean(data)
    std = np.std(data)

    if std == 0:
        print(f"⚠️ STD=0, skipping: {input_path}")
        return None

    normalized_data = (data - mean) / std
    stream[0].data = normalized_data.astype(np.float32)

    base, ext = os.path.splitext(input_path)
    output_path = f"{base}_zscore{ext}"

    # Αν υπάρχει ήδη, μην το ξαναγράψεις
    if os.path.exists(output_path):
        print(f"⏭️ Already exists: {output_path}")
        return output_path

    stream.write(output_path, format='MSEED')
    return output_path



def zscore_normalize_LQT(min_snr, max_ps_duration, min_event_duration,
                         depth_min, depth_max):
    """
    Normalize ONLY L/Q/T mseed files inside folders like:

        <BASE_DIR>/<year>/<event>/<station>/<min_snr>_<max_ps_duration>_<min_event_duration>_(depthmin-depthmax)/

    For each L, Q, T file → produce *_zscore.mseed
    """

    from main import BASE_DIR

    # Pattern of the folder where LQT files exist
    folder_name = f"{min_snr}_{max_ps_duration}_{min_event_duration}_({depth_min}-{depth_max})"

    valid_suffixes = ("_L.mseed", "_Q.mseed", "_T.mseed")

    print(f"\n🔍 Searching for L/Q/T files in folders named: {folder_name}")

    for year in os.listdir(BASE_DIR):
        year_path = os.path.join(BASE_DIR, year)
        if not os.path.isdir(year_path):
            continue

        for event_name in os.listdir(year_path):
            event_path = os.path.join(year_path, event_name)
            if not os.path.isdir(event_path):
                continue

            for station_name in os.listdir(event_path):
                station_path = os.path.join(event_path, station_name)
                if not os.path.isdir(station_path):
                    continue

                # --- Look for the LQT folder ---
                lqt_path = os.path.join(station_path, folder_name)
                if not os.path.isdir(lqt_path):
                    continue

                # --- Scan for L/Q/T mseed files ---
                for fname in os.listdir(lqt_path):
                    if fname.endswith(valid_suffixes):
                        full_path = os.path.join(lqt_path, fname)

                        try:
                            output = z_score_normalize_mseed(full_path)
                            if output:
                                print(f"✅ Normalized: {output}")
                        except Exception as e:
                            print(f"❌ Failed: {full_path} — {e}")

    print("\n🎉 Completed Z-score normalization for all L/Q/T files.\n")
