import os
import json

def load_json(path):
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def plot_peak_segmentation_duration_distribution(bin_size: float = 5.0):
    """
    Υπολογίζει και σχεδιάζει την κατανομή (ραβδόγραμμα)
    των duration_of_peak_segment τιμών ΜΟΝΟ για τα Z κανάλια (π.χ. HHZ, BHZ, EHZ)
    από το αρχείο boundaries.json και το αποθηκεύει στο Logs/station-duration-distribution.png
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from main import LOG_DIR

    # --- Ανάγνωση αρχείου ---
    json_path = os.path.join(LOG_DIR, "boundaries.json")
    if not os.path.exists(json_path):
        print(f"❌ Δεν βρέθηκε το αρχείο: {json_path}")
        return

    data = load_json(json_path)
    durations = []

    # --- Διασχίζουμε τη δομή: έτος → event → σταθμό → κανάλι ---
    for year, events in data.items():
        if year == "total_nof_stations":
            continue
        for event_name, stations in events.items():
            for station_name, channels in stations.items():
                if not isinstance(channels, dict):
                    continue

                # Μόνο τα κανάλια Z (HHZ, BHZ, EHZ)
                for ch_name, ch_info in channels.items():
                    if not isinstance(ch_info, dict):
                        continue
                    if not ch_name.endswith("Z"):
                        continue

                    dur = ch_info.get("peak_segment_duration_time")
                    if dur is None:
                        continue

                    try:
                        durations.append(float(dur))
                    except ValueError:
                        continue

    if not durations:
        print("❌ Δεν βρέθηκαν τιμές duration_of_peak_segment για κανάλια Z")
        return

    # --- Bins ---
    max_value = max(durations)
    bins = np.arange(0, max_value + bin_size, bin_size)

    # --- Ραβδόγραμμα ---
    plt.figure(figsize=(10, 6))
    counts, bins, patches = plt.hist(durations, bins=bins, color="teal", edgecolor="black", alpha=0.8)

    plt.title("Distribution Peak Segmentation Duration (only Z channels)", fontsize=14, fontweight="bold")
    plt.xlabel("Duration(sec)", fontsize=12)
    plt.ylabel("Nof Stations", fontsize=12)
    plt.grid(axis="y", linestyle="--", alpha=0.6)

    # Προσθήκη labels πάνω από κάθε μπάρα
    for c, p in zip(counts, patches):
        if c > 0:
            plt.text(p.get_x() + p.get_width() / 2, c, f"{int(c)}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()

    # --- Αποθήκευση ---
    output_png = os.path.join(LOG_DIR, "station-duration-distribution.png")
    plt.savefig(output_png, dpi=200)
    print(f"💾 Histogram stored at {output_png}")

    plt.show()

def plot_clean_event_duration_distribution(bin_size: float = 5.0):
    """
    Υπολογίζει και σχεδιάζει την κατανομή (ραβδόγραμμα)
    των event_duration_time τιμών ΜΟΝΟ για τα Z κανάλια (HHZ, BHZ, EHZ)
    από το αρχείο boundaries.json και το αποθηκεύει στο Logs/clean-event-duration-distribution.png
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from main import LOG_DIR

    # --- Ανάγνωση αρχείου ---
    json_path = os.path.join(LOG_DIR, "boundaries.json")
    if not os.path.exists(json_path):
        print(f"❌ Δεν βρέθηκε το αρχείο: {json_path}")
        return

    data = load_json(json_path)
    durations = []

    # --- Διασχίζουμε τη δομή: year → event → station → channel ---
    for year, events in data.items():
        if year == "total_nof_stations":
            continue  # skip global key

        for event_name, stations in events.items():
            for station_name, channels in stations.items():
                if not isinstance(channels, dict):
                    continue

                # μόνο τα Z κανάλια
                for ch_name, ch_info in channels.items():
                    if not isinstance(ch_info, dict):
                        continue
                    if not ch_name.endswith("Z"):
                        continue

                    dur = ch_info.get("clean_event_duration_time")
                    if dur is None:
                        continue

                    try:
                        durations.append(float(dur))
                    except ValueError:
                        continue

    if not durations:
        print("❌ Δεν βρέθηκαν τιμές event_duration_time για κανάλια Z")
        return

    # --- Bins ---
    max_value = max(durations)
    bins = np.arange(0, max_value + bin_size, bin_size)

    # --- Ραβδόγραμμα ---
    plt.figure(figsize=(10, 6))
    counts, bins, patches = plt.hist(durations, bins=bins, color="purple", edgecolor="black", alpha=0.8)

    plt.title("Distribution of Clean Event Duration (Z channels only)", fontsize=14, fontweight="bold")
    plt.xlabel("Duration (seconds)", fontsize=12)
    plt.ylabel("Number of stations", fontsize=12)
    plt.grid(axis="y", linestyle="--", alpha=0.6)

    # αριθμοί πάνω από κάθε μπάρα
    for c, p in zip(counts, patches):
        if c > 0:
            plt.text(p.get_x() + p.get_width() / 2, c, f"{int(c)}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()

    # --- Αποθήκευση ---
    output_png = os.path.join(LOG_DIR, "clean-event-duration-distribution.png")
    plt.savefig(output_png, dpi=200)
    print(f"💾 Αποθηκεύτηκε στο {output_png}")

    plt.show()

def plot_snr_distribution(bin_size: float = 3.0):
    """
    Υπολογίζει και σχεδιάζει την κατανομή (ραβδόγραμμα)
    των minimum_station_snr τιμών από το αρχείο boundaries.json
    και το αποθηκεύει στο Logs/snr-distribution.png
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from main import LOG_DIR

    # --- Διαδρομή αρχείου ---
    json_path = os.path.join(LOG_DIR, "boundaries.json")

    # --- Έλεγχος ύπαρξης ---
    if not os.path.exists(json_path):
        print(f"❌ Δεν βρέθηκε το αρχείο: {json_path}")
        return

    # --- Ανάγνωση δεδομένων ---
    data = load_json(json_path)
    snr_values = []

    # --- Δομή: έτος → γεγονός → σταθμός ---
    for year, events in data.items():
        if not isinstance(events, dict):
            continue
        for event_name, stations in events.items():
            if not isinstance(stations, dict):
                continue
            for station_name, station_info in stations.items():
                if not isinstance(station_info, dict):
                    continue

                # Αν υπάρχει τιμή minimum_station_snr στο επίπεδο σταθμού
                min_snr = station_info.get("minimum_station_snr")
                if min_snr is None:
                    continue

                try:
                    snr_values.append(float(min_snr))
                except (TypeError, ValueError):
                    continue

    if not snr_values:
        print("❌ Δεν βρέθηκαν τιμές minimum_station_snr στο boundaries.json")
        return

    # --- Δημιουργία bins ---
    max_value = max(snr_values)
    bins = np.arange(0, max_value + bin_size, bin_size)

    # --- Ραβδόγραμμα ---
    plt.figure(figsize=(10, 6))
    counts, bins, patches = plt.hist(
        snr_values, bins=bins, color="orange", edgecolor="black", alpha=0.8
    )

    plt.title("Distribution SNR per station", fontsize=14, fontweight="bold")
    plt.xlabel("SNR (value per station)", fontsize=12)
    plt.ylabel("Nof Stations", fontsize=12)
    plt.grid(axis="y", linestyle="--", alpha=0.6)

    # Προσθήκη labels πάνω από κάθε μπάρα
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
    output_png = os.path.join(LOG_DIR, "snr-distribution.png")
    plt.savefig(output_png, dpi=200)
    print(f"💾 Histogram stored at {output_png}")

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
    #plot_clean_event_duration_distribution()
    #plot_peak_segmentation_duration_distribution()
    #plot_snr_distribution()
    plot_depth_distribution()