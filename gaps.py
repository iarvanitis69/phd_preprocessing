import os
import json
from obspy import read


def find_gaps_in_file(file_path):
    """
    Βρίσκει gaps σε ένα .mseed αρχείο και επιστρέφει λίστα με μεταδεδομένα για κάθε gap.
    """
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

def find_files_for_gaps():
    """
    Σαρώνει όλους τους υποφακέλους και βρίσκει gaps σε αρχεία .mseed
    """
    all_gaps = {}

    from main import BASE_DIR
    for root, _, files in os.walk(BASE_DIR):
        for file in files:
            if file.endswith(".mseed"):
                full_path = os.path.join(root, file)

                # Χρήσιμο σχετικό path για το JSON log
                from main import BASE_DIR
                rel_path = os.path.relpath(full_path, BASE_DIR)

                print("🔍 Ελέγχεται για gaps: " + rel_path)
                gaps = find_gaps_in_file(full_path)
                if gaps:
                    all_gaps[rel_path] = gaps

    from main import GAPS_FILE
    print(f"💾 Αποθήκευση στο αρχείο: {GAPS_FILE}")
    with open(GAPS_FILE, "w", encoding="utf-8") as f:
        json.dump(all_gaps, f, indent=2, ensure_ascii=False)

    print("✅ Ολοκληρώθηκε με επιτυχία η αναζήτηση και καταγραφή gaps.")
