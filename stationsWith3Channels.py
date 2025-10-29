import os
import json
import shutil

def count_mseed_files(mseed_path):
    if not os.path.isdir(mseed_path):
        return 0
    return len([f for f in os.listdir(mseed_path) if f.endswith(".mseed")])

def find_stations_with_nofChannelsL3_json_file():
    """
    Εντοπίζει σταθμούς με ≠ 3 .mseed αρχεία και τους προσθέτει στο JSON αρχείο
    Logs/nofChannelsL3.json, χωρίς να ξαναγράφει το αρχείο από την αρχή.
    Διατηρεί συνολικό μετρητή "Account of stations with channels lower than 3".
    """
    from main import BASE_DIR

    logs_dir = os.path.join(BASE_DIR, "Logs")
    os.makedirs(logs_dir, exist_ok=True)
    output_path = os.path.join(logs_dir, "nofChannelsL3.json")

    # Διάβασε υπάρχον JSON αν υπάρχει
    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = {"Account of stations with channels lower than 3": 0}

    total_count = data.get("Account of stations with channels lower than 3", 0)

    # Κεντρική δομή με τα events
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
                if not os.path.isdir(station_path) or station.lower() == "info.txt":
                    continue

                mseed_path = os.path.join(station_path, "mseed")
                file_count = count_mseed_files(mseed_path)

                if file_count != 3:
                    if event not in data:
                        data[event] = {}

                    # Αν ο σταθμός δεν υπάρχει ήδη στο JSON για το event, προσθέτουμε
                    if station not in data[event]:
                        data[event][station] = {
                            "nofChannels": file_count
                        }
                        total_count += 1

    # Ενημέρωσε το συνολικό μετρητή
    data["Account of stations with channels lower than 3"] = total_count

    # Αποθήκευση JSON
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"[✔] Ενημερώθηκε: {output_path} ({total_count} σταθμοί συνολικά)")

def delete_stations_with_nofChannels_l3():
    """
    Διαγράφει όλους τους σταθμούς που έχουν λιγότερα από 3 .mseed αρχεία,
    βάσει του αρχείου Logs/nofChannelsL3.json. Αν διαγραφούν όλοι οι σταθμοί ενός event,
    διαγράφεται και ο φάκελος του event.
    """
    from main import BASE_DIR

    logs_dir = os.path.join(BASE_DIR, "Logs")
    json_path = os.path.join(logs_dir, "nofChannelsL3.json")

    if not os.path.exists(json_path):
        print(f"[✘] Δεν βρέθηκε το αρχείο: {json_path}")
        return

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    count_deleted_stations = 0
    count_deleted_events = 0

    for event, stations in data.items():
        if event == "Account of stations with channels lower than 3":
            continue

        event_path = None

        for station in stations:
            # Βρες το path του σταθμού
            year = event[:4]
            event_path = os.path.join(BASE_DIR, year, event)
            station_path = os.path.join(event_path, station)

            if os.path.isdir(station_path):
                try:
                    shutil.rmtree(station_path)
                    print(f"[ΔΙΑΓΡΑΦΗ] {year}/{event}/{station}")
                    count_deleted_stations += 1
                except Exception as e:
                    print(f"[ΣΦΑΛΜΑ] Δεν διαγράφηκε {station_path}: {e}")

        # Αν ο φάκελος του event υπάρχει και είναι πλέον άδειος → διαγραφή
        if event_path and os.path.isdir(event_path) and len(os.listdir(event_path)) == 0:
            try:
                shutil.rmtree(event_path)
                print(f"[ΔΙΑΓΡΑΦΗ EVENT] {year}/{event} (κενό)")
                count_deleted_events += 1
            except Exception as e:
                print(f"[ΣΦΑΛΜΑ] Δεν διαγράφηκε ο φάκελος event: {event_path}: {e}")

    print(f"\n[✔] Ολοκληρώθηκε: Διαγράφηκαν {count_deleted_stations} σταθμοί και {count_deleted_events} event φάκελοι.")

