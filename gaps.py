import os
import json
from obspy import read


def find_gaps_in_file(file_path):
    """Βρίσκει gaps σε ένα .mseed αρχείο και επιστρέφει λίστα με μεταδεδομένα για κάθε gap."""
    try:
        stream = read(file_path)
        gaps = stream.get_gaps()
        result = []

        for gap in gaps:
            network, station, location, channel = gap[:4]
            starttime, endtime = gap[4], gap[5]
            missing_samples = gap[6]

            gap_type = "Interpolation" if missing_samples <= 10 else "NaN values"

            prev_time = str(starttime - 0.01)
            next_time = str(endtime + 0.01)

            result.append({
                "network": network,
                "station": station,
                "location": location,
                "channel": channel,
                "starttime": str(starttime),
                "endtime": str(endtime),
                "prev_time": prev_time,
                "next_time": next_time,
                "missing_samples": missing_samples,
                "gap_type": gap_type
            })

        return result
    except Exception as e:
        print(f"⚠️ Σφάλμα στο αρχείο {file_path}: {e}")
        return []

def update_excluded_stations(event_name, station_name, logs_dir):
    """Προσθέτει τον σταθμό στο excluded_stations.json με λόγο gaps, εφόσον δεν υπάρχει ήδη."""
    excluded_path = os.path.join(logs_dir, "excluded_stations.json")
    if os.path.exists(excluded_path):
        with open(excluded_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = {}

    if event_name not in data:
        data[event_name] = {}

    if station_name not in data[event_name]:
        data[event_name][station_name] = {
            "reason": "Gaps found"
        }

        with open(excluded_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"⚠️ Εξαιρέθηκε ο σταθμός {station_name} από το συμβάν {event_name} λόγω gaps.")

def load_excluded_stations(logs_dir):
    """Φορτώνει το excluded_stations.json ως λεξικό {event: {station: {...}}}"""
    excluded_path = os.path.join(logs_dir, "excluded_stations.json")
    if os.path.exists(excluded_path):
        with open(excluded_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def find_files_for_gaps():
    """Σαρώνει όλους τους υποφακέλους και βρίσκει gaps σε αρχεία .mseed."""
    all_gaps = {}

    from main import BASE_DIR, GAPS_FILE
    logs_dir = os.path.join(BASE_DIR, "Logs")
    os.makedirs(logs_dir, exist_ok=True)

    # 🔹 Φόρτωσε το excluded_stations.json μία φορά
    excluded = load_excluded_stations(logs_dir)

    for root, _, files in os.walk(BASE_DIR):
        for file in files:
            if file.endswith(".mseed"):
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, BASE_DIR)

                # 🔹 Βρες Event και Station από τη διαδρομή
                parts = rel_path.split(os.sep)
                if len(parts) < 3:
                    continue  # δεν έχει event/station info

                event = parts[-3]
                station = parts[-2]

                # 🔸 Αν αυτός ο σταθμός είναι ήδη excluded, προχώρα στο επόμενο
                if event in excluded and station in excluded[event]:
                    print(f"⏭️ Παράλειψη excluded station: {event}/{station}")
                    continue

                print("🔍 Ελέγχεται για gaps: " + rel_path)
                gaps = find_gaps_in_file(full_path)
                if gaps:
                    all_gaps[rel_path] = gaps
                    update_excluded_stations(event, station, logs_dir)

    print(f"💾 Αποθήκευση στο αρχείο: {GAPS_FILE}")
    with open(GAPS_FILE, "w", encoding="utf-8") as f:
        json.dump(all_gaps, f, indent=2, ensure_ascii=False)

    print("✅ Ολοκληρώθηκε με επιτυχία η αναζήτηση και καταγραφή gaps.")

