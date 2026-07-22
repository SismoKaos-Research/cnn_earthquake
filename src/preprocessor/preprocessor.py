import re
from datetime import datetime

import numpy as np
import numpy.ma as _nma
import scipy.interpolate as _sci_interp
from obspy import UTCDateTime, read

from src.preprocessor.config import settings


def _gap_duration_sec(g, st_full):
    tr_match = st_full.select(channel=g[3])
    fs_val = (
        tr_match[0].stats.sampling_rate
        if tr_match
        else (st_full[0].stats.sampling_rate if len(st_full) > 0 else 100.0)
    )
    return g[7] / fs_val


def parse_file_date(mseed_file, st_full):
    match = re.search(r"_(\d{8})_", mseed_file.name)
    if match:
        date_str = match.group(1)
        for date_fmt in [(4,8, 2,4, 0,2), (0,4, 4,6, 6,8)]:
            try:
                y_s, y_e, m_s, m_e, d_s, d_e = date_fmt
                return datetime(int(date_str[y_s:y_e]), int(date_str[m_s:m_e]), int(date_str[d_s:d_e]))
            except ValueError:
                continue
    return st_full[0].stats.starttime.datetime


def _interpolate_nans(data, mask):
    valid = np.where(~np.isnan(data))[0]
    if len(valid) > 3:
        f = _sci_interp.interp1d(
            valid, data[valid], kind="cubic", bounds_error=False, fill_value="extrapolate"
        )
        data[np.where(mask)[0]] = f(np.where(mask)[0])
    return data


# Modules 
def _manage_gaps(st_full):
    gaps = st_full.get_gaps(min_gap=-1)
    actual_gaps = [g for g in gaps if g[7] > 0] if gaps else []
    
    large_gaps = [g for g in actual_gaps if _gap_duration_sec(g, st_full) >= settings.GAP_THRESHOLD]
    small_gaps = [g for g in actual_gaps if _gap_duration_sec(g, st_full) < settings.GAP_THRESHOLD]
    has_large_gap = len(large_gaps) > 0

    print(f"[INFO] Küçük gap (<{settings.GAP_THRESHOLD}s): {len(small_gaps)} | Büyük gap (≥{settings.GAP_THRESHOLD}s): {len(large_gaps)}")

    if not actual_gaps:
        st_full.merge()
        return st_full

    st_full.merge(fill_value=np.nan)

    for tr in st_full:
        data = np.ma.filled(tr.data.astype(float), np.nan) if np.ma.is_masked(tr.data) else np.array(tr.data, dtype=float)

        if not has_large_gap:
            nan_mask = np.isnan(data)
            if nan_mask.any():
                data = _interpolate_nans(data, nan_mask)
            tr.data = data
        else:
            big_gap_mask = np.zeros(len(data), dtype=bool)
            tr_start, tr_fs = tr.stats.starttime, tr.stats.sampling_rate

            for lg in large_gaps:
                if lg[1] == tr.stats.station and lg[3] == tr.stats.channel:
                    gi_start = max(0, int((lg[4] - tr_start) * tr_fs))
                    gi_end = min(len(data), int((lg[5] - tr_start) * tr_fs) + 1)
                    if gi_start < gi_end:
                        big_gap_mask[gi_start:gi_end] = True

            small_nan = np.isnan(data) & ~big_gap_mask
            if small_nan.any():
                data = _interpolate_nans(data, small_nan)

            tr.data = np.ma.array(data, mask=big_gap_mask)
            
    return st_full


def _apply_base_filters(tr):
    tr.detrend("demean")
    tr.detrend("linear")
    tr.filter(
        "bandpass", freqmin=settings.FREQMIN, freqmax=settings.FREQMAX, corners=4, zerophase=True
    )


def _process_trace(wtr, res_tr, win_nan_map, real_decimation_factor, min_seg_len):
    has_gap = wtr.id in win_nan_map

    if not has_gap:
        tmp = wtr.copy()
        _apply_base_filters(tmp)
        if real_decimation_factor > 1:
            tmp.decimate(factor=real_decimation_factor, no_filter=True)
            tmp.detrend("linear")
            
        res_tr.data = tmp.data
        res_tr.stats.sampling_rate = tmp.stats.sampling_rate
        return

    nan_mask = win_nan_map[wtr.id]
    raw = wtr.data.copy().astype(float)
    
    out_n = len(raw) // real_decimation_factor if real_decimation_factor > 1 else len(raw)
    out = np.full(out_n, np.nan, dtype=float)

    pad = np.concatenate([[False], ~nan_mask, [False]])
    d_arr = np.diff(pad.astype(np.int8))
    
    starts = np.where(d_arr == 1)[0]
    ends = np.where(d_arr == -1)[0]

    for ss, se in zip(starts, ends):
        if (se - ss) < min_seg_len:
            continue
            
        tmp = wtr.copy()
        tmp.data = raw[ss:se].copy()
        tmp.stats.starttime = wtr.stats.starttime + ss / wtr.stats.sampling_rate
        
        try:
            _apply_base_filters(tmp)
        except Exception:
            continue
            
        dec = tmp.data[::real_decimation_factor] if real_decimation_factor > 1 else tmp.data
        o_start = ss // real_decimation_factor
        o_end = min(o_start + len(dec), out_n)
        
        if o_end - o_start > 0:
            out[o_start:o_end] = dec[: o_end - o_start]
            
    res_tr.data = out
    res_tr.stats.sampling_rate = settings.Fs


def _save_window_to_ml_array(st_decimated, window_start_utc):
    date_folder = window_start_utc.datetime.strftime("%Y_%m_%d")
    timestamp = window_start_utc.datetime.strftime('%Y%m%d_%H%M%S')

    components = ["Z", "N", "E"]
    
    expected_length = int(settings.PREPROCESS_WINDOW_SEC * settings.Fs)
    
    ml_array = np.zeros((3, expected_length), dtype=np.float32)
    
    available_traces = 0

    for i, comp in enumerate(components):
        try:
            tr = st_decimated.select(component=comp)[0]
            
            data_out = _nma.filled(tr.data, np.nan) if _nma.is_masked(tr.data) else tr.data
            
            # Replace NaNs with 0.0 to prevent gradient corruption in PyTorch
            data_out = np.nan_to_num(data_out, nan=0.0)
            
            actual_len = min(len(data_out), expected_length)
            ml_array[i, :actual_len] = data_out[:actual_len]
            
            available_traces += 1
        except IndexError:
            continue

    # Skip saving if no channels were found
    if available_traces == 0:
        return 0

    comp_dir = settings.DATA_ROOT / date_folder
    comp_dir.mkdir(parents=True, exist_ok=True)
    
    npy_path = comp_dir / f"{timestamp}_ZNE.npy"
    np.save(npy_path, ml_array)
    
    return 1



# main function
def run_mseed_preprocessing():
    print("\n" + "=" * 50)
    print("AŞAMA 1: MSEED ÖN İŞLEME (GAP, FİLTRE, DOWNSAMPLE)")
    print("=" * 50)

    if not settings.MSEED_INPUT_DIR.exists():
        print(f"[ATLANDI] MSEED dizini bulunamadı: {settings.MSEED_INPUT_DIR}")
        return False

    # 1. Directory Parsing Logic
    # Find all .mseed and .miniseed files recursively in the directory
    mseed_files = list(settings.MSEED_INPUT_DIR.rglob("*.mseed"))
    mseed_files.extend(list(settings.MSEED_INPUT_DIR.rglob("*.miniseed")))

    if not mseed_files:
        print(f"[UYARI] {settings.MSEED_INPUT_DIR} içinde MSEED dosyası bulunamadı.")
        return False

    print(f"[INFO] Toplam {len(mseed_files)} adet MSEED dosyası bulundu. İşlem başlıyor...")

    processing_start_time = datetime.now()
    
    # Master Log Setup
    log_dir = settings.DATA_ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    gap_log_path = log_dir / f"gap_report_{processing_start_time.strftime('%Y%m%d_%H%M%S')}.txt"

    if settings.SAVE_REPORT:
        with open(gap_log_path, "w", encoding="utf-8") as glog:
            glog.write(f"TOPLU GAP RAPORU — Başlangıç: {processing_start_time}\n")
            glog.write(f"Hedef FS: {settings.Fs} Hz | Bandpass: {settings.FREQMIN}–{settings.FREQMAX} Hz | Gap Eşiği: {settings.GAP_THRESHOLD} sn\n")
            glog.write("=" * 60 + "\n\n")

    total_files_created_all = 0

    # 2. Iterate over all discovered files
    for file_idx, mseed_file in enumerate(mseed_files, 1):
        print(f"\n[INFO] [{file_idx}/{len(mseed_files)}] Dosya okunuyor: {mseed_file.name}")
        
        if settings.SAVE_REPORT:
            with open(gap_log_path, "a", encoding="utf-8") as glog:
                glog.write(f"--- Dosya: {mseed_file.name} ---\n")

        try:
            st_full = read(str(mseed_file))
        except Exception as e:
            print(f"[HATA] {mseed_file.name} okunamadı: {e}")
            continue

        for tr_raw in st_full:
            if tr_raw.data.dtype != np.float64:
                tr_raw.data = tr_raw.data.astype(np.float64)

        real_fs = st_full[0].stats.sampling_rate
        real_decimation_factor = max(1, int(real_fs / settings.Fs))

        # Handle Gaps
        st_full = _manage_gaps(st_full)

        file_date = parse_file_date(mseed_file, st_full)
        day_start = UTCDateTime(file_date.year, file_date.month, file_date.day, 0, 0, 0)
        day_end = day_start + 86400
        num_windows = int((day_end - day_start) / settings.PREPROCESS_WINDOW_SEC)
        
        min_seg_len = max(20, int(3.0 / settings.FREQMIN * real_fs))
        total_files_created = 0

        for window_idx in range(num_windows):
            window_start_utc = day_start + (window_idx * settings.PREPROCESS_WINDOW_SEC)
            window_end_utc = window_start_utc + settings.PREPROCESS_WINDOW_SEC
            st_window = st_full.copy().slice(starttime=window_start_utc, endtime=window_end_utc)

            if len(st_window) == 0 or len(st_window[0].data) < 100:
                continue

            st_proc = st_window.copy()
            win_nan_map = {}
            for wtr in st_proc:
                wd = np.ma.filled(wtr.data.astype(float), np.nan) if np.ma.is_masked(wtr.data) else np.array(wtr.data, dtype=float)
                wnan = np.isnan(wd)
                if wnan.any():
                    win_nan_map[wtr.id] = wnan
                wtr.data = wd

            st_decimated = st_proc.copy()
            
            for wtr in st_proc:
                res_tr = st_decimated.select(id=wtr.id)[0]
                _process_trace(wtr, res_tr, win_nan_map, real_decimation_factor, min_seg_len)

            total_files_created += _save_window_to_ml_array(st_decimated, window_start_utc)

        total_files_created_all += total_files_created
        print(f"[INFO] {mseed_file.name} tamamlandı. ({total_files_created} NPY oluşturuldu)")

    print("\n" + "=" * 50)
    print(f"[BAŞARILI] Tüm ön işleme tamamlandı. Toplam {total_files_created_all} NPY dosyası {settings.DATA_ROOT} içine oluşturuldu.")
    print("=" * 50)
    return True

if __name__ == "__main__":
    run_mseed_preprocessing()
