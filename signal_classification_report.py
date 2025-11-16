def classify_signals(min_snr=5, max_ps_duration=30.0, min_clean_event_duration=30.0,
                      depth_min=1.0, depth_max=24.0):
    """
    Classification of HHZ signals into:
        • Training eligible
        • B1: High SNR & PS too long
        • B2: Low SNR
        • B3: High SNR & Clean event too short
        • B4: High SNR & Depth outside valid range

    Compatible with NEW boundaries_HHZ.json format.
    """

    import os
    from utils import load_json, save_json
    from main import LOG_DIR, BASE_DIR

    boundaries_path = os.path.join(LOG_DIR, "boundaries_HHZ.json")
    if not os.path.exists(boundaries_path):
        print(f"❌ boundaries_HHZ.json not found at {boundaries_path}")
        return

    data = load_json(boundaries_path)

    # -----------------------------
    # Counters
    # -----------------------------
    total = 0
    aic_fail = 0

    B1_too_long = 0
    B2_low_snr = 0
    B3_clean_too_short = 0
    B4_depth_bad = 0

    TRAIN = 0

    # =============================================================
    # TRAVERSE boundaries_HHZ.json
    # =============================================================
    for year, events in data.items():

        if year == "total_nof_stations":
            continue

        for event_name, stations in events.items():

            # ---- LOAD Depth ----
            info_path = os.path.join(BASE_DIR, str(year), event_name, "info.json")
            depth_km = None

            if os.path.exists(info_path):
                try:
                    info = load_json(info_path)
                    depth_km = (
                        info.get("depth_km")
                        or info.get("Depth_km")
                        or info.get("depth-km")
                        or info.get("Depth-km")
                    )
                    depth_km = float(depth_km)
                except:
                    depth_km = None

            for station_name, station_info in stations.items():

                total += 1

                if not isinstance(station_info, dict):
                    continue

                # -----------------------------
                # Missing AIC?
                # -----------------------------
                if station_info.get("start_of_event_HHZ_idx") is None:
                    aic_fail += 1
                    continue

                # -----------------------------
                # Extract parameters
                # -----------------------------
                snr = station_info.get("minimum_station_snr")
                ps_dur = station_info.get("peak_segment_duration_HHZ_time")
                ce_dur = station_info.get("clean_event_duration_HHZ_time")

                try:
                    snr = float(snr)
                    ps_dur = float(ps_dur)
                    ce_dur = float(ce_dur)
                except:
                    continue

                # -----------------------------
                # Classification Rules
                # -----------------------------

                # B2: SNR < min_snr
                if snr < min_snr:
                    B2_low_snr += 1
                    continue

                # B1: High SNR but PS too long
                if ps_dur > max_ps_duration:
                    B1_too_long += 1
                    continue

                # B3: High SNR but clean_event too short
                if ce_dur < min_clean_event_duration:
                    B3_clean_too_short += 1
                    continue

                # Depth filtering
                if depth_km is None or not (depth_min <= depth_km <= depth_max):
                    B4_depth_bad += 1
                    continue

                # TRAINING SET
                TRAIN += 1

    # =============================================================
    # PREPARE REPORT DICT
    # =============================================================
    report = {
        "total_station_signals": total,
        "AIC_failure_signals": aic_fail,
        f"NOT_USED_PS_too_long_(PS>{max_ps_duration})": B1_too_long,
        f"NOT_USED_SNR_below_{min_snr}": B2_low_snr,
        f"NOT_USED_clean_event_too_short_(<{min_clean_event_duration})": B3_clean_too_short,
        f"NOT_USED_Depth_outside_{depth_min}-{depth_max}km": B4_depth_bad,
        "TRAINING_SET": TRAIN,
        "parameters": {
            "min_snr": min_snr,
            "max_ps_duration": max_ps_duration,
            "min_clean_event_duration": min_clean_event_duration,
            "depth_min": depth_min,
            "depth_max": depth_max
        }
    }

    # =============================================================
    # SAVE JSON REPORT
    # =============================================================
    output_path = os.path.join(LOG_DIR, "classification_report.json")
    save_json(output_path, report)

    # =============================================================
    # PRINT REPORT
    # =============================================================
    print("\n📊 *** SIGNAL CLASSIFICATION REPORT ***")
    print(f"Total station signals: {total}")
    print(f"AIC_failure signals: {aic_fail}")
    print(f"⚠ NOT USED SET : SNR ≥ {min_snr} & PS_duration_time > {max_ps_duration} sec : {B1_too_long}")
    print(f"⚠ NOT USED SET : SNR < {min_snr} : {B2_low_snr}")
    print(f"⚠ NOT USED SET : SNR ≥ {min_snr} & clean_event_duration_time < {min_clean_event_duration} sec : {B3_clean_too_short}")
    print(f"⚠ NOT USED SET : SNR ≥ {min_snr} & (Depth <{depth_min} Km or Depth >{depth_max} Km) : {B4_depth_bad}")
    print("-------------------------------------------------------------------------------------------------------------------")
    print(f"✔ TRAINING SET : SNR ≥ {min_snr} & PS_duration_time ≤ {max_ps_duration} sec & clean_event_duration_time ≥ {min_clean_event_duration} sec & {depth_min}Km<=Depth<={depth_max}Km : {TRAIN}\n")
    print(f"💾 Classification report saved → {output_path}")

    return report

# ==========================================================
if __name__ == "__main__":
    classify_signals()