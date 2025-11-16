def find_cutting_info(
        min_snr: float,
        max_ps_duration: float,
        min_event_duration: float,
        depth_min: float,
        depth_max: float):
    """
    Unified function that produces BOTH:
        • PS-FIXED JSON
        • PS-VARIANT JSON

    Compatible with the NEW boundaries_HHZ.json format, where:
        - HHN, HHE store only total_signal info
        - HHZ segmentation info is stored at station-level with names:
            peak_segment_duration_HHZ_time
            clean_event_duration_HHZ_time
            minimum_station_snr
    """

    import os
    from collections import OrderedDict
    from main import LOG_DIR, BASE_DIR
    from utils import load_json, save_json

    boundaries_path = os.path.join(LOG_DIR, "boundaries_HHZ.json")

    if not os.path.exists(boundaries_path):
        print(f"❌ File not found: {boundaries_path}")
        return

    boundaries_json = load_json(boundaries_path)

    psfixed_json = {}
    psvariant_json = {}

    print("\n🔍 Running Find Cutting Info (new boundaries_HHZ.json format)...")

    # ------------------------------------------------------------
    # Traverse year → event → station
    # ------------------------------------------------------------
    for year, events in boundaries_json.items():

        if year == "total_nof_stations":
            continue
        if not isinstance(events, dict):
            continue

        for event_name, stations in events.items():

            # -----------------------------
            # Load depth from info.json
            # -----------------------------
            depth_km = None
            info_path = os.path.join(BASE_DIR, year, event_name, "info.json")

            if os.path.exists(info_path):
                info = load_json(info_path)
                depth_km = (
                    info.get("depth_km")
                    or info.get("Depth_km")
                    or info.get("depth-km")
                    or info.get("Depth-km")
                )
                try:
                    depth_km = float(depth_km)
                except:
                    depth_km = None

            if depth_km is None:
                continue

            # Depth filtering
            if not (depth_min <= depth_km <= depth_max):
                continue

            # ------------------------------------------------------------
            # Evaluate each station
            # ------------------------------------------------------------
            for station_name, station_info in stations.items():

                if not isinstance(station_info, dict):
                    continue

                # ------------------------
                # SNR
                # ------------------------
                snr_value = station_info.get("minimum_station_snr")
                if snr_value is None:
                    continue
                snr_value = float(snr_value)

                if snr_value < min_snr:
                    continue

                # ------------------------
                # Required HHZ fields
                # ------------------------
                ps_time = station_info.get("peak_segment_duration_HHZ_time")
                ce_time = station_info.get("clean_event_duration_HHZ_time")

                if ps_time is None:
                    continue

                try:
                    ps_time = float(ps_time)
                except:
                    continue

                # ------------------------
                # PS-VARIANT condition:
                # only peak duration
                # ------------------------
                is_variant_valid = (ps_time <= max_ps_duration)

                # ------------------------
                # PS-FIXED condition:
                # peak OK + clean-event OK
                # ------------------------
                is_fixed_valid = False
                if is_variant_valid and ce_time is not None:
                    try:
                        ce_time = float(ce_time)
                        if ce_time >= min_event_duration:
                            is_fixed_valid = True
                    except:
                        pass

                # ------------------------
                # SAVE INTO PS-FIXED JSON
                # ------------------------
                if is_fixed_valid:
                    year_dict = psfixed_json.setdefault(year, {})
                    event_dict = year_dict.setdefault(event_name, {})
                    event_dict[station_name] = station_info

                # ------------------------
                # SAVE INTO PS-VARIANT JSON
                # ------------------------
                if is_variant_valid:
                    year_dict = psvariant_json.setdefault(year, {})
                    event_dict = year_dict.setdefault(event_name, {})
                    event_dict[station_name] = station_info

    # ------------------------------------------------------------------
    # COUNT stations (HHZ only)
    # ------------------------------------------------------------------
    def count_hhz(struct):
        cnt = 0
        for y, events in struct.items():
            for ev, stations in events.items():
                for st, info in stations.items():
                    if "peak_segment_duration_HHZ_time" in info:
                        cnt += 1
        return cnt

    total_psfixed = count_hhz(psfixed_json)
    total_psvariant = count_hhz(psvariant_json)

    # ------------------------------------------------------------------
    # REORDER JSON to include total count first
    # ------------------------------------------------------------------
    def reorder(struct, total):
        out = OrderedDict()
        out["total_nof_stations"] = total
        for y, ev in struct.items():
            out[y] = ev
        return out

    ordered_psfixed = reorder(psfixed_json, total_psfixed)
    ordered_psvariant = reorder(psvariant_json, total_psvariant)

    # ------------------------------------------------------------------
    # SAVE OUTPUT JSON FILES
    # ------------------------------------------------------------------
    file_fixed = (
        f"StationsForPsFixed_{min_snr}_{max_ps_duration}_{min_event_duration}_({depth_min}-{depth_max}).json"
    )

    file_variant = (
        f"StationsForPsVariantsAndCleanEvents_{min_snr}_{max_ps_duration}_({depth_min}-{depth_max}).json"
    )

    save_json(os.path.join(LOG_DIR, file_fixed), ordered_psfixed)
    save_json(os.path.join(LOG_DIR, file_variant), ordered_psvariant)

    print("\n💾 CREATED FILES:")
    print(f"   ✔ PS-FIXED   → {file_fixed}")
    print(f"   ✔ PS-VARIANT → {file_variant}")
    print(f"📦 PS-FIXED   stations: {total_psfixed}")
    print(f"📦 PS-VARIANT stations: {total_psvariant}\n")

    return ordered_psfixed, ordered_psvariant



def create_cutting_signals(
        min_snr: float,
        max_ps_duration: float,
        min_event_duration: float,
        depth_min: float,
        depth_max: float):
    """
    Creates:
       • PS-Fixed  (.mseed)
       • PS-Variant (.mseed)
       • WholeEvent (.mseed)

    Based on the JSON files generated by find_cutting_info():
         StationsForPsFixed_...
         StationsForPsVariantsAndCleanEvents_...

    Exact algorithm:
        1) Load PS-Fixed JSON
        2) For all Z-channels:
              • load original _demeanDetrend_IC_BPF.mseed (UNCHANGED)
              • extract:
                    - PS-Fixed:
                         same peak-length for ALL stations
                    - PS-Variant:
                         each station keeps its own peak duration
                    - WholeEvent:
                         from start_idx → end_event_idx

    All generated files are stored inside a single folder per station:
        <min_snr>__<max_ps_duration>__<min_event_duration>__<depth_min>-<depth_max>
    """

    import os
    from obspy import read
    from main import LOG_DIR, BASE_DIR
    from utils import load_json

    # ------------------------------------------------------------
    # JSON FILENAMES (constructed exactly as in find_cutting_info)
    # ------------------------------------------------------------
    file_fixed = (
        f"StationsForPsFixed_{min_snr}_{max_ps_duration}_"
        f"{min_event_duration}_({depth_min}-{depth_max}).json"
    )

    file_variant = (
        f"StationsForPsVariantsAndCleanEvents_{min_snr}_{max_ps_duration}_"
        f"({depth_min}-{depth_max}).json"
    )

    path_fixed = os.path.join(LOG_DIR, file_fixed)
    path_variant = os.path.join(LOG_DIR, file_variant)

    if not os.path.exists(path_fixed):
        print(f"❌ Missing file: {path_fixed}")
        return

    if not os.path.exists(path_variant):
        print(f"❌ Missing file: {path_variant}")
        return

    db_fixed = load_json(path_fixed)
    db_variant = load_json(path_variant)

    print("\n🔍 Starting create_cutting_signals...")

    # ------------------------------------------------------------
    # Determine GLOBAL max PS duration (PS-Fixed)
    # ------------------------------------------------------------
    print("🔎 Computing global PS-Fixed peak length ...")
    max_peak_seconds = 0.0

    for year, events in db_fixed.items():
        if year == "total_nof_stations":
            continue

        if not isinstance(events, dict):
            continue

        for event_name, stations in events.items():
            if not isinstance(stations, dict):
                continue

            for station_name, channels in stations.items():
                if not isinstance(channels, dict):
                    continue

                # for ch_name, ch_info in channels.items():
                #     if not (isinstance(ch_info, dict) and ch_name.endswith("Z")):
                #         continue
                ps = channels.get("peak_segment_duration_HHZ_time")
                if ps is None:
                   continue
                try:
                   ps = float(ps)
                except:
                   continue
                max_peak_seconds = max(max_peak_seconds, ps)

    print(f"📌 Global maximum PS-Fixed duration: {max_peak_seconds:.3f} sec")

    # ------------------------------------------------------------
    # MAIN CUTTING PROCEDURE (PS-Fixed + PS-Variant + WholeEvent)
    # ------------------------------------------------------------
    def cut_files_from_json(json_db, mode):
        """
        mode ∈ {"PSfixed", "PSvariant"}
        """
        for year, events in json_db.items():

            if year == "total_nof_stations":
                continue
            if not isinstance(events, dict):
                continue

            for event_name, stations in events.items():
                if not isinstance(stations, dict):
                    continue

                for station_name, channels in stations.items():

                    if not isinstance(channels, dict):
                        continue

                    station_path = os.path.join(BASE_DIR, str(year), event_name, station_name)
                    if not os.path.exists(station_path):
                        continue

                    # 📁 common output directory
                    folder_name = (
                        f"{min_snr}_{max_ps_duration}_"
                        f"{min_event_duration}_({depth_min}-{depth_max})"
                    )
                    output_dir = os.path.join(station_path, folder_name)
                    os.makedirs(output_dir, exist_ok=True)

                    # find original IC_BPF file
                    orig_file = None
                    for fname in os.listdir(station_path):
                        if fname.endswith("_demeanDetrend_IC_BPF.mseed"):
                            orig_file = os.path.join(station_path, fname)
                            break

                    if orig_file is None:
                        print(f"⚠️ No IC_BPF file found for {station_name} in {station_path}")
                        continue

                    # read waveform
                    try:
                        st = read(orig_file)
                    except Exception as e:
                        print(f"⚠️ Cannot read {orig_file}: {e}")
                        continue

                    tr = st[0]
                    sr = tr.stats.sampling_rate

                    # ------------------------------------------------
                    # INDICES FROM JSON
                    # ------------------------------------------------
                    start_idx = channels.get("start_of_event_HHZ_idx")
                    end_peak_idx = channels.get("end_of_peak_segment_HHZ_idx")
                    end_event_idx = channels.get("end_of_event_HHZ_idx")

                    if start_idx is None or end_peak_idx is None or end_event_idx is None:
                        continue

                    # ====================================================
                    #  A)  PS-FIXED
                    # ====================================================
                    if mode == "PSfixed":

                        duration_samples = int(round(max_peak_seconds * sr))

                        output_name = orig_file.replace(
                            "_demeanDetrend_IC_BPF.mseed",
                            "_demeanDetrend_IC_BPF_PSfixed.mseed"
                        )
                        output_name = os.path.join(output_dir, os.path.basename(output_name))

                        # ----- SKIP IF FILE ALREADY EXISTS -----
                        if os.path.exists(output_name):
                            print(f"⏭️ Skip (exists): {output_name}")
                            continue

                        seg = tr.copy().data[start_idx:start_idx + duration_samples]
                        if len(seg) == 0:
                            continue

                        tr2 = tr.copy()
                        tr2.data = seg
                        tr2.write(output_name, format="MSEED")
                        print(f"✔ Created: {output_name}")

                    # ====================================================
                    #  B)  PS-VARIANT
                    # ====================================================
                    if mode == "PSvariant":

                        output_name = orig_file.replace(
                            "_demeanDetrend_IC_BPF.mseed",
                            "_demeanDetrend_IC_BPF_PSvariant.mseed"
                        )
                        output_name = os.path.join(output_dir, os.path.basename(output_name))

                        # ----- SKIP IF FILE ALREADY EXISTS -----
                        if os.path.exists(output_name):
                            print(f"⏭️ Skip (exists): {output_name}")
                            continue

                        seg = tr.copy().data[start_idx:end_peak_idx]
                        if len(seg) == 0:
                            continue

                        tr2 = tr.copy()
                        tr2.data = seg
                        tr2.write(output_name, format="MSEED")
                        print(f"✔ Created: {output_name}")

                        # ====================================================
                        #  C)  WHOLE EVENT (only in PSvariant mode)
                        # ====================================================
                        whole_name = orig_file.replace(
                            "_demeanDetrend_IC_BPF.mseed",
                            "_demeanDetrend_IC_BPF_WholeEvent.mseed"
                        )
                        whole_name = os.path.join(output_dir, os.path.basename(whole_name))

                        # ----- SKIP IF FILE ALREADY EXISTS -----
                        if os.path.exists(whole_name):
                            print(f"⏭️ Skip (exists): {whole_name}")
                            continue

                        seg_event = tr.copy().data[start_idx:end_event_idx]
                        if len(seg_event) == 0:
                            continue

                        tr3 = tr.copy()
                        tr3.data = seg_event
                        tr3.write(whole_name, format="MSEED")
                        print(f"✔ Created: {whole_name}")

    # ------------------------------------------------------------
    # RUN CUTTING
    # ------------------------------------------------------------
    cut_files_from_json(db_fixed, mode="PSfixed")
    cut_files_from_json(db_variant, mode="PSvariant")

    print("\n✅ Finished creating:")
    print("   • PSfixed files")
    print("   • PSvariant files")
    print("   • WholeEvent files")
    print(f"   All under folders named:{min_snr}_{max_ps_duration}_{min_event_duration}_({depth_min}-{depth_max})\n")


# ==========================================================
if __name__ == "__main__":

    #find_cutting_info(5, 30, 30, 1,24)
    create_cutting_signals(5, 30, 30, 1,24)