import os
import json
import shutil

def count_mseed_files(station_path):
    """Μετρά τα .mseed αρχεία απευθείας μέσα στον φάκελο του σταθμού."""
    if not os.path.isdir(station_path):
        return 0
    return len([f for f in os.listdir(station_path) if f.endswith(".mseed")])

def find_stations_with_nofChannelsL3():
    """
    Εντοπίζει σταθμούς με <3 .mseed αρχεία και τους προσθέτει στο JSON:
    Logs/nofChannelsL3.json, χωρίς να ξαναγράφει όλο το αρχείο.
    """
    #from main import BASE_EVENTS_DIR as BASE_DIR

    from main import BASE_DIR
    logs_dir = os.path.join(BASE_DIR, "Logs")
    os.makedirs(logs_dir, exist_ok=True)
    output_path = os.path.join(logs_dir, "nofChannelsL3.json")

    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = {"Count of stations with channels lower than 3": 0}

    total_count = data.get("Count of stations with channels lower than 3", 0)

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

                file_count = count_mseed_files(station_path)

                if file_count < 3:
                    if event not in data:
                        data[event] = {}

                    if station not in data[event]:
                        data[event][station] = {"nofChannels": file_count}
                        total_count += 1

    data["Count of stations with channels lower than 3"] = total_count

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"[✔] Ενημερώθηκε: {output_path} ({total_count} σταθμοί συνολικά)")


def delete_stations_with_nofChannels_l3():
    """Διαγράφει σταθμούς με <3 .mseed βάσει του JSON."""

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
        if event == "Count of stations with channels lower than 3":
            continue

        year = event[:4]
        event_path = os.path.join(BASE_DIR, year, event)

        for station in stations:
            station_path = os.path.join(event_path, station)
            if os.path.isdir(station_path):
                try:
                    shutil.rmtree(station_path)
                    print(f"[ΔΙΑΓΡΑΦΗ] {year}/{event}/{station}")
                    count_deleted_stations += 1
                except Exception as e:
                    print(f"[ΣΦΑΛΜΑ] Δεν διαγράφηκε {station_path}: {e}")

        if event_path and os.path.isdir(event_path) and len(os.listdir(event_path)) == 0:
            try:
                shutil.rmtree(event_path)
                print(f"[ΔΙΑΓΡΑΦΗ EVENT] {year}/{event} (κενό)")
                count_deleted_events += 1
            except Exception as e:
                print(f"[ΣΦΑΛΜΑ] Δεν διαγράφηκε ο φάκελος event: {event_path}: {e}")

    print(f"\n[✔] Ολοκληρώθηκε: Διαγράφηκαν {count_deleted_stations} σταθμοί και {count_deleted_events} event φάκελοι.")
