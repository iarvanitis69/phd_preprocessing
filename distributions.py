import os
import json


def plot_peak_segmentation_duration_distribution(bin_size: float = 5.0):
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from main import LOG_DIR
    from utils import load_json

    json_path = os.path.join(LOG_DIR, "boundaries_HHZ.json")
    if not os.path.exists(json_path):
        print(f"❌ boundaries_HHZ.json not found: {json_path}")
        return

    data = load_json(json_path)
    durations = []

    # --- Scan years → events → stations ---
    for year, events in data.items():
        if year == "total_nof_stations":
            continue

        for event_name, stations in events.items():
            for station_name, station_info in stations.items():
                if not isinstance(station_info, dict):
                    continue

                dur = station_info.get("peak_segment_duration_HHZ_time")
                if dur is None:
                    continue
                try:
                    durations.append(float(dur))
                except:
                    continue

    if not durations:
        print("❌ No peak_segment_duration_HHZ_time found!")
        return

    max_value = max(durations)
    bins = np.arange(0, max_value + bin_size, bin_size)

    plt.figure(figsize=(10, 6))
    counts, bins, patches = plt.hist(
        durations, bins=bins, color="teal", edgecolor="black", alpha=0.8
    )

    plt.title("Peak Segmentation Duration Distribution (HHZ only)")
    plt.xlabel("Duration (sec)")
    plt.ylabel("Nof Stations")
    plt.grid(axis="y", linestyle="--", alpha=0.6)

    for c, p in zip(counts, patches):
        if c > 0:
            plt.text(
                p.get_x() + p.get_width() / 2, c,
                f"{int(c)}", ha="center", va="bottom", fontsize=9
            )

    plt.tight_layout()

    output_png = os.path.join(LOG_DIR, "peak-segment-duration-distribution.png")
    plt.savefig(output_png, dpi=200)
    print(f"💾 Saved at {output_png}")
    plt.show()

def plot_clean_event_duration_distribution(bin_size: float = 5.0):
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from main import LOG_DIR
    from utils import load_json

    json_path = os.path.join(LOG_DIR, "boundaries_HHZ.json")
    if not os.path.exists(json_path):
        print(f"❌ boundaries_HHZ.json not found!")
        return

    data = load_json(json_path)
    durations = []

    for year, events in data.items():
        if year == "total_nof_stations":
            continue

        for event_name, stations in events.items():
            for station_name, station_info in stations.items():
                if not isinstance(station_info, dict):
                    continue

                dur = station_info.get("clean_event_duration_HHZ_time")
                if dur is None:
                    continue

                try:
                    durations.append(float(dur))
                except:
                    continue

    if not durations:
        print("❌ No clean_event_duration_HHZ_time found!")
        return

    max_value = max(durations)
    bins = np.arange(0, max_value + bin_size, bin_size)

    plt.figure(figsize=(10, 6))
    counts, bins, patches = plt.hist(
        durations, bins=bins, color="purple", edgecolor="black", alpha=0.8
    )

    plt.title("Clean Event Duration Distribution (HHZ only)")
    plt.xlabel("Duration (sec)")
    plt.ylabel("Nof Stations")
    plt.grid(axis="y", linestyle="--", alpha=0.6)

    for c, p in zip(counts, patches):
        if c > 0:
            plt.text(
                p.get_x() + p.get_width() / 2, c,
                f"{int(c)}", ha="center", va="bottom", fontsize=9
            )

    plt.tight_layout()

    output_png = os.path.join(LOG_DIR, "clean-event-duration-distribution.png")
    plt.savefig(output_png, dpi=200)
    print(f"💾 Saved at {output_png}")

    plt.show()

def plot_snr_distribution(bin_size: float = 3.0):
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from main import LOG_DIR
    from utils import load_json

    json_path = os.path.join(LOG_DIR, "boundaries_HHZ.json")

    if not os.path.exists(json_path):
        print(f"❌ boundaries_HHZ.json not found!")
        return

    data = load_json(json_path)
    snr_values = []

    for year, events in data.items():
        if year == "total_nof_stations":
            continue

        for event_name, stations in events.items():
            for station_name, station_info in stations.items():
                if not isinstance(station_info, dict):
                    continue

                min_snr = station_info.get("minimum_station_snr")
                if min_snr is None:
                    continue

                try:
                    snr_values.append(float(min_snr))
                except:
                    continue

    if not snr_values:
        print("❌ No minimum_station_snr values found!")
        return

    max_value = max(snr_values)
    bins = np.arange(0, max_value + bin_size, bin_size)

    plt.figure(figsize=(10, 6))
    counts, bins, patches = plt.hist(
        snr_values, bins=bins, color="orange", edgecolor="black", alpha=0.8
    )

    plt.title("SNR Distribution per Station")
    plt.xlabel("SNR")
    plt.ylabel("Nof Stations")
    plt.grid(axis="y", linestyle="--", alpha=0.6)

    for c, p in zip(counts, patches):
        if c > 0:
            plt.text(
                p.get_x() + p.get_width() / 2, c,
                f"{int(c)}", ha="center", va="bottom"
            )

    plt.tight_layout()

    output_png = os.path.join(LOG_DIR, "snr-distribution.png")
    plt.savefig(output_png, dpi=200)
    print(f"💾 Saved at {output_png}")
    plt.show()


def plot_depth_distribution(bin_size: float = 1.0):
    """
    Υπολογίζει και σχεδιάζει την κατανομή (ραβδόγραμμα)
    των depth_km τιμών για ΟΛΑ τα events.

    Για κάθε event ανοίγει:
        Events/<YEAR>/<EVENT>/info.json

    Η τιμή βάθους βρίσκεται στο πεδίο:
        "depth_km"

    Το γράφημα αποθηκεύεται ως:
        Logs/DepthDistribution.png
    """

    import os
    import json
    import numpy as np
    import matplotlib.pyplot as plt
    from main import BASE_DIR, LOG_DIR

    depth_values = []

    # --- Σάρωση όλων των ετών ---
    for year in os.listdir(BASE_DIR):
        year_path = os.path.join(BASE_DIR, year)
        if not os.path.isdir(year_path):
            continue

        # --- Σάρωση όλων των events ---
        for event_name in os.listdir(year_path):
            event_path = os.path.join(year_path, event_name)
            if not os.path.isdir(event_path):
                continue

            # Αναζήτηση του info.json στο event
            info_path = os.path.join(event_path, "info.json")
            if not os.path.exists(info_path):
                continue

            # --- Ανάγνωση depth από info.json ---
            try:
                with open(info_path, "r", encoding="utf-8") as f:
                    info = json.load(f)

                depth = info.get("depth_km")

                if depth is None:
                    continue

                depth_values.append(float(depth))

            except Exception as e:
                print(f"⚠️ Αποτυχία ανάγνωσης {info_path}: {e}")
                continue

    if not depth_values:
        print("❌ Δεν βρέθηκαν τιμές depth_km σε κανένα info.json")
        return

    # --- Δημιουργία bins ---
    max_value = max(depth_values)
    bins = np.arange(0, max_value + bin_size, bin_size)

    # --- Ραβδόγραμμα ---
    plt.figure(figsize=(10, 6))
    counts, bins, patches = plt.hist(
        depth_values, bins=bins, color="steelblue", edgecolor="black", alpha=0.85
    )

    plt.title("Depth Distribution of All Events", fontsize=14, fontweight="bold")
    plt.xlabel("Depth (km)", fontsize=12)
    plt.ylabel("Number of Events", fontsize=12)
    plt.grid(axis="y", linestyle="--", alpha=0.6)

    for c, p in zip(counts, patches):
        if c > 0:
            plt.text(
                p.get_x() + p.get_width() / 2,
                c,
                f"{int(c)}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    plt.tight_layout()

    # --- Αποθήκευση ---
    output_png = os.path.join(LOG_DIR, "DepthDistribution.png")
    plt.savefig(output_png, dpi=200)
    print(f"💾 Depth histogram stored at: {output_png}")

    plt.show()

# ==========================================================
if __name__ == "__main__":
    plot_clean_event_duration_distribution()
    plot_peak_segmentation_duration_distribution()
    plot_snr_distribution()
    plot_depth_distribution()