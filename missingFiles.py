import os
import json

def count_mseed_files(station_path):
    """Μετρά τα .mseed αρχεία στον φάκελο του σταθμού."""
    if not os.path.isdir(station_path):
        return 0
    return len([f for f in os.listdir(station_path) if f.endswith(".mseed")])

def has_stationxml_file(station_path):
    """Ελέγχει αν υπάρχει αρχείο .xml στον φάκελο του σταθμού."""
    return any(f.endswith(".xml") for f in os.listdir(station_path))

def find_stations_with_issues():
    """
    Εντοπίζει σταθμούς με <3 .mseed αρχεία ή χωρίς StationXML και ενημερώνει:
    1. Logs/missingFiles.json
    2. Logs/excluded_stations.json (με reason)
    """
    from main import BASE_DIR  # π.χ. /path/to/Seismic/Events/
    logs_dir = os.path.join(BASE_DIR, "Logs")
    os.makedirs(logs_dir, exist_ok=True)

    # Αρχεία εξόδου
    path_missing_files = os.path.join(logs_dir, "missingFiles.json")
    path_excluded = os.path.join(logs_dir, "excluded_stations.json")

    # Ανάγνωση υπαρχόντων δεδομένων
    data_missing = {}
    if os.path.exists(path_missing_files):
        with open(path_missing_files, "r", encoding="utf-8") as f:
            data_missing = json.load(f)

    data_excluded = {}
    if os.path.exists(path_excluded):
        with open(path_excluded, "r", encoding="utf-8") as f:
            data_excluded = json.load(f)

    if "COUNT" not in data_excluded:
        data_excluded["COUNT"] = 0

    # Βρόχος αναζήτησης
    for year in sorted(os.listdir(BASE_DIR)):
        year_path = os.path.join(BASE_DIR, year)
        if not os.path.isdir(year_path) or year.lower() == "logs":
            continue

        for event in sorted(os.listdir(year_path)):
            event_path = os.path.join(year_path, event)
            if not os.path.isdir(event_path):
                continue

            for station in sorted(os.listdir(event_path)):
                station_path = os.path.join(event_path, station)
                if not os.path.isdir(station_path):
                    continue

                reason = None
                file_count = count_mseed_files(station_path)
                has_xml = has_stationxml_file(station_path)

                if file_count < 3:
                    reason = "channels lower than 3"
                elif not has_xml:
                    reason = "missing StationXML file"

                if reason:
                    # --- missingFiles.json ---
                    if event not in data_missing:
                        data_missing[event] = {}
                    if station not in data_missing[event]:
                        data_missing[event][station] = {
                            "reason": reason
                        }

                    # --- excluded_stations.json ---
                    if event not in data_excluded:
                        data_excluded[event] = {}
                    if station not in data_excluded[event]:
                        data_excluded[event][station] = {
                            "reason": reason
                        }
                        data_excluded["COUNT"] += 1

    # Αποθήκευση
    with open(path_missing_files, "w", encoding="utf-8") as f:
        json.dump(data_missing, f, indent=2, ensure_ascii=False)

    with open(path_excluded, "w", encoding="utf-8") as f:
        json.dump(data_excluded, f, indent=2, ensure_ascii=False)

    print(f"[✔] Ενημερώθηκαν:")
    print(f"     → {path_missing_files}")
    print(f"     → {path_excluded} (Σύνολο excluded COUNT = {data_excluded['COUNT']})")
