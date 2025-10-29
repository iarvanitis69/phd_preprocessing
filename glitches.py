import os
import json
import numpy as np
from obspy import read, Trace
import shutil

from concurrent.futures import ProcessPoolExecutor, as_completed

def extract_event_info(event_folder_name: str):
    """
    Π.χ. '20100507T041515_36.68_25.71_15.0km_M3.4'
    Επιστρέφει dict με πληροφορίες σεισμικού γεγονότος.
    """
    try:
        parts = event_folder_name.split("_")
        origin_time = parts[0]
        lat = parts[1]
        lon = parts[2]
        depth = parts[3].replace("km", "")
        mag = parts[4].replace("M", "")
        return {
            "event_folder": event_folder_name,
            "origin_time": origin_time,
            "latitude": float(lat),
            "longitude": float(lon),
            "depth_km": float(depth),
            "magnitude": float(mag)
        }
    except Exception:
        return {
            "event_folder": event_folder_name,
            "origin_time": None,
            "latitude": None,
            "longitude": None,
            "depth_km": None,
            "magnitude": None
        }

def find_glitches(trace: Trace, threshold, window_size: int = 2):
    """
    Επιστρέφει λίστα από glitches που εντοπίστηκαν στο trace.
    """
    data = trace.data.astype(float)
    max_val = np.max(np.abs(data))
    if max_val == 0:
        return []

    data /= max_val
    glitches = []

    for i in range(len(data) - 2 * window_size):
        win1 = data[i:i + window_size]
        win2 = data[i + window_size:i + 2 * window_size]
        d1 = np.diff(win1)
        d2 = np.diff(win2)

        peak_rise = np.max(d1)
        peak_fall = np.min(d2)

        # Άνοδος ακολουθούμενη από κάθοδο
        if peak_rise > threshold and peak_fall < -threshold:
            glitches.append({
                "start_index": i,
                "end_index": i + 2 * window_size,
                "channel": trace.stats.channel,
                "station": trace.stats.station,
                "start_time": str(trace.stats.starttime + i / trace.stats.sampling_rate),
                "end_time": str(trace.stats.starttime + (i + 2 * window_size) / trace.stats.sampling_rate),
                "peak_rise": round(peak_rise, 4),
                "peak_fall": round(peak_fall, 4)
            })
    return glitches

def find_files_for_glitches_parallel(threshold: float = 1.0, max_workers: int = 4):
    """
    Εκτελεί παράλληλη επεξεργασία γεγονότων και γράφει άμεσα τα glitches στο JSON.
    """
    from main import BASE_DIR
    print(f"🚀 Έναρξη σάρωσης στο: {BASE_DIR}")

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for year in sorted(os.listdir(BASE_DIR)):
            year_path = os.path.join(BASE_DIR, year)
            if not os.path.isdir(year_path) or year == "Logs":
                continue

            for event in sorted(os.listdir(year_path)):
                event_path = os.path.join(year_path, event)
                if not os.path.isdir(event_path):
                    continue

                futures.append(executor.submit(process_single_event, event_path, threshold))

        for f in as_completed(futures):
            print(f.result())

    print("💾 Όλα τα glitches έχουν ήδη γραφτεί στο Logs/glitches.json.")

def process_single_event(event_path: str, threshold: float):
    """
    Επεξεργάζεται ένα σεισμικό γεγονός και γράφει κάθε glitch
    άμεσα στο glitches.json με το νέο format.
    """
    event_info = extract_event_info(os.path.basename(event_path))
    event_name = event_info["event_folder"]
    for station in sorted(os.listdir(event_path)):
        station_path = os.path.join(event_path, station)
        mseed_path = os.path.join(station_path, "mseed")
        if not os.path.isdir(mseed_path):
            continue

        for fname in os.listdir(mseed_path):
            if not fname.endswith(".mseed"):
                continue

            full_path = os.path.join(mseed_path, fname)
            try:
               st = read(full_path)
               for tr in st:
                   glitches = find_glitches(tr, threshold=threshold)
                   if glitches:
                        for g in glitches:
                            g["file"] = fname
                        station_id = f"{tr.stats.network}.{tr.stats.station}"
                        append_to_json_file(event_name, station_id, tr.stats.channel, glitches)
                        print(
                            f"📈 {event_name} | {tr.stats.station} | {tr.stats.channel} | {len(glitches)} glitches")
            except Exception as e:
                print(f"⚠️ Σφάλμα στο αρχείο {fname}: {e}")

    return f"✅ Ολοκληρώθηκε: {event_name}"

def append_to_json_file(event_name, station, channel, glitches):
    """
    Προσθέτει τα glitches ενός καναλιού μέσα στο Logs/glitches.json
    στο σωστό format:
    event → station → channel → {count, glitches: [...]}
    """
    from main import BASE_DIR
    logs_path = os.path.join(BASE_DIR, "Logs")
    os.makedirs(logs_path, exist_ok=True)
    output_file = os.path.join(logs_path, "glitches.json")
    import multiprocessing
    lock = multiprocessing.Manager().Lock()
    with lock:
        # Διάβασμα υπάρχοντος JSON (αν υπάρχει)
        if os.path.exists(output_file):
            with open(output_file, "r", encoding="utf-8") as f:
               try:
                   data = json.load(f)
               except json.JSONDecodeError:
                   data = {}
        else:
            data = {}

        # --- Ενημέρωση δομής ---
        if event_name not in data:
            data[event_name] = {}
        if station not in data[event_name]:
            data[event_name][station] = {}
        if channel not in data[event_name][station]:
            data[event_name][station][channel] = {"count": 0, "glitches": []}

        # --- Προσθήκη νέων glitches ---
        data[event_name][station][channel]["glitches"].extend(glitches)
        data[event_name][station][channel]["count"] += len(glitches)
        # --- Εγγραφή στο αρχείο ---
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

def delete_files_with_glitches():
    """
    Διαγράφει όλους τους σταθμούς που έχουν glitches σε τουλάχιστον ένα channel.
    Αν μετά τη διαγραφή δεν υπάρχει κανένας σταθμός στο event, διαγράφεται και το event.
    """
    from main import BASE_DIR

    glitches_path = os.path.join(BASE_DIR, "Logs", "glitches.json")
    if not os.path.exists(glitches_path):
        print(f"[✘] Δεν βρέθηκε το αρχείο: {glitches_path}")
        return

    with open(glitches_path, "r", encoding="utf-8") as f:
        try:
            glitches_data = json.load(f)
        except json.JSONDecodeError:
            print("[✘] Το αρχείο glitches.json είναι άδειο ή κατεστραμμένο.")
            return

    deleted_stations = 0
    deleted_events = 0

    for event, stations in glitches_data.items():
        year = event[:4]
        event_path = os.path.join(BASE_DIR, year, event)

        for station in stations.keys():
            station_path = os.path.join(event_path, station)
            if os.path.isdir(station_path):
                try:
                    shutil.rmtree(station_path)
                    print(f"[ΔΙΑΓΡΑΦΗ] {year}/{event}/{station} (λόγω glitch)")
                    deleted_stations += 1
                except Exception as e:
                    print(f"[ΣΦΑΛΜΑ] Δεν διαγράφηκε ο σταθμός {station_path}: {e}")

        # Έλεγχος αν απέμεινε κάτι στο event
        if os.path.isdir(event_path) and len(os.listdir(event_path)) == 0:
            try:
                shutil.rmtree(event_path)
                print(f"[ΔΙΑΓΡΑΦΗ EVENT] {year}/{event} (κενό μετά από διαγραφές)")
                deleted_events += 1
            except Exception as e:
                print(f"[ΣΦΑΛΜΑ] Δεν διαγράφηκε το event {event_path}: {e}")

    print(f"\n[✔] Ολοκληρώθηκε: {deleted_stations} σταθμοί διαγράφηκαν, {deleted_events} κενά events διαγράφηκαν.")
