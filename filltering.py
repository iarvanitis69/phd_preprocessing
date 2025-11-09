import os
import json
from obspy import read, Stream


def load_excluded_stations(logs_dir: str) -> dict:
    path = os.path.join(logs_dir, "excluded_stations.json")
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def is_excluded(excluded: dict, event: str, station: str) -> bool:
    return event in excluded and station in excluded[event]


def guess_event_name(file_path: str) -> str:
    parts = os.path.normpath(file_path).split(os.sep)
    if "Events" in parts:
        idx = parts.index("Events")
        if idx + 2 < len(parts):
            return parts[idx + 2]
    return os.path.splitext(os.path.basename(file_path))[0]


def iter_mseed_files(root: str):
    for dirpath, _, filenames in os.walk(root):
        if "Logs" in dirpath:
            continue
        for fn in filenames:
            if fn.endswith("_demean_detrend_IC.mseed"):
                yield os.path.join(dirpath, fn)


def apply_bandpass(stream: Stream, freqmin: float = 0.5, freqmax: float = 45.0, corners: int = 4) -> Stream:
    return stream.copy().filter("bandpass", freqmin=freqmin, freqmax=freqmax, corners=corners, zerophase=True)


def process_file(path: str, excluded: dict, output_suffix="_BPF"):
    try:
        st = read(path)
    except Exception as e:
        print(f"⚠️ Αποτυχία ανάγνωσης {path}: {e}")
        return

    event = guess_event_name(path)
    filtered = Stream()

    for tr in st:
        net = getattr(tr.stats, "network", "XX") or "XX"
        sta = getattr(tr.stats, "station", "UNK") or "UNK"
        station = f"{net}.{sta}"

        if is_excluded(excluded, event, station):
            print(f"⛔ Παράλειψη excluded σταθμού: {event}/{station}")
            continue

        try:
            filtered += tr.copy().filter("bandpass", freqmin=0.5, freqmax=45.0, corners=4, zerophase=True)
        except Exception as e:
            print(f"⚠️ Σφάλμα bandpass {station}: {e}")

    if filtered:
        new_path = path.replace("_demean_detrend_IC.mseed", f"_demean_detrend_IC{output_suffix}.mseed")
        filtered.write(new_path, format="MSEED")
        print(f"✅ Αποθηκεύτηκε: {new_path}")
    else:
        print(f"⚠️ Κανένα ίχνος για αποθήκευση στο {path}")


def filter_all_files():
    from main import BASE_DIR, LOG_DIR
    excluded = load_excluded_stations(LOG_DIR)

    for path in iter_mseed_files(BASE_DIR):
        process_file(path, excluded)


if __name__ == "__main__":
    filter_all_files()
