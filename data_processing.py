import pandas as pd
import numpy as np
import os
from scipy.signal import find_peaks, butter, filtfilt
from ahrs.filters import Madgwick
from ahrs.common.orientation import q2euler
from datetime import datetime

input_dir = './dataset/'
output_dir = './cleaned_data/'
os.makedirs(output_dir, exist_ok=True)
csv_files = [f for f in os.listdir(input_dir) if f.endswith('.csv')]
print("Found", len(csv_files), "csv files")

# desired columns for output (no spo2 because dataset doesn't have)
desired_columns = ['time', 'tempC', 'accX', 'accY', 'accZ',
                   'gyroX', 'gyroY', 'gyroZ', 'angleX', 'angleY', 'angleZ', 'hr']

# sensitivities (tùy chỉnh nếu bạn biết exact scale)
DEFAULT_ACC_LSB_PER_G = 16384.0   # ±2g accCoef = 0.0000611     1 / 16384   → ±2g (16384 LSB/g)
DEFAULT_GYRO_LSB_PER_DEG = 131.0  # gyroCoef = 0.007633         1 / 131     → ±250°/s (131 LSB/°/s)

# helpers
def parse_time_column(df, time_col='time'):
    # try multiple formats; convert to pandas datetime
    df[time_col] = pd.to_datetime(df[time_col])
    return df

def estimate_fs(df, time_col='time'):
    # return sampling freq in Hz (median inverse dt)
    times = pd.to_datetime(df[time_col]).astype('int64') // 1_000_000_000  # seconds
    dts = np.diff(times)
    # if seconds are integer (low precision) fallback to using milliseconds if available in original
    if len(dts) == 0 or np.median(dts) == 0:
        # try with nanoseconds precision
        times_ns = pd.to_datetime(df[time_col]).astype('int64')  # ns
        dts_ns = np.diff(times_ns) / 1e9
        dt = np.median(dts_ns) if len(dts_ns)>0 else 0.002
    else:
        dt = np.median(dts)
    fs = 1.0 / dt if dt > 0 else 500.0
    return fs

def detect_and_normalize_acc(df, acc_cols=['accX','accY','accZ'], acc_lsb_per_g=DEFAULT_ACC_LSB_PER_G):
    if not all(c in df.columns for c in acc_cols):
        return df, "missing"
    acc_absmax = df[acc_cols].abs().values.max()
    # Heuristics:
    # - if values ~ 5..20 -> likely m/s^2 -> convert to g by /9.80665
    # - if values ~ 0..16 and often ~9.8 -> likely m/s^2
    # - if values ~ 0..4 -> likely already in g
    # - if values >> 50 -> likely raw counts -> divide by LSB/g -> g
    if acc_absmax > 50:
        # raw counts
        df[acc_cols] = df[acc_cols].astype(float) / acc_lsb_per_g
        source = "raw_counts->g"
    elif acc_absmax > 4.0:
        # m/s^2 -> convert to g
        df[acc_cols] = df[acc_cols].astype(float) / 9.80665
        source = "m/s2->g"
    else:
        # already in g
        df[acc_cols] = df[acc_cols].astype(float)
        source = "already_g"
    return df, source

def detect_and_normalize_gyro(df, gyro_cols=['gyroX','gyroY','gyroZ'], gyro_lsb_per_deg=DEFAULT_GYRO_LSB_PER_DEG):
    if not all(c in df.columns for c in gyro_cols):
        return df, "missing"
    gyro_absmax = df[gyro_cols].abs().values.max()
    # heuristics:
    # - if values < 3 -> likely rad/s -> to deg/s multiply
    # - if values between 3 and 100 -> likely deg/s already
    # - if values >> 200 -> likely raw counts -> divide by LSB -> deg/s
    if gyro_absmax > 200:
        # raw counts
        df[gyro_cols] = df[gyro_cols].astype(float) / gyro_lsb_per_deg
        source = "raw_counts->deg/s"
    elif gyro_absmax < 3.0:
        # rad/s -> deg/s
        df[gyro_cols] = df[gyro_cols].astype(float) * 57.29577951308232
        source = "rad/s->deg/s"
    else:
        df[gyro_cols] = df[gyro_cols].astype(float)
        source = "already_deg/s"
    return df, source

def compute_angles_from_imu(df, acc_cols=['accX','accY','accZ'], gyro_cols=['gyroX','gyroY','gyroZ']):
    # Madgwick: expects gyr in (deg/s) or rad/s? ahrs lib expects deg/s typically when updateIMU called
    # pass gyro in deg/s (as we've normalized above) and acc in g.
    acc_data = df[acc_cols].values
    gyr_data = df[gyro_cols].values
    madgwick = Madgwick()
    N = len(df)
    quat = np.zeros((N,4))
    if N==0:
        return df
    quat[0] = np.array([1.0,0.0,0.0,0.0])
    for i in range(1,N):
        try:
            quat[i] = madgwick.updateIMU(quat[i-1], gyr=gyr_data[i], acc=acc_data[i])
        except Exception:
            quat[i] = quat[i-1]
    eulers = np.array([q2euler(q) for q in quat])  # returns radians (roll,pitch,yaw) > convert to degrees
    eulers_deg = np.degrees(eulers)
    df['angleX'] = eulers_deg[:,0]
    df['angleY'] = eulers_deg[:,1]
    df['angleZ'] = eulers_deg[:,2]
    return df

# bandpass filter for ECG
def bandpass_filter(x, fs, low=5.0, high=30.0, order=3):
    nyq = 0.5*fs
    lown = low/nyq
    highn = high/nyq
    b,a = butter(order, [lown := lown, highn], btype='band')
    y = filtfilt(b,a,x)
    return y

def calculate_hr_from_ecg(df, ecg_column='ecg', time_column='time', fs=None):
    # returns hr_series aligned with df index (bpm)
    hr_series = pd.Series(np.nan, index=df.index)
    if ecg_column not in df.columns:
        return hr_series
    if fs is None:
        fs = estimate_fs(df, time_column)
    ecg = df[ecg_column].astype(float).values
    if len(ecg) < 3:
        return hr_series
    # bandpass to emphasize QRS
    try:
        ecg_f = bandpass_filter(ecg, fs, low=5.0, high=30.0, order=3)
    except Exception:
        ecg_f = ecg - np.mean(ecg)
    # dynamic peak detection
    min_distance = int(0.3 * fs)  # min 200-300ms between beats
    height_thr = np.percentile(ecg_f, 90)  # threshold based on percentile
    peaks, props = find_peaks(ecg_f, distance=min_distance, height=height_thr, prominence=np.std(ecg_f)*0.5)
    if len(peaks) < 2:
        # relax threshold and try again
        peaks, props = find_peaks(ecg_f, distance=min_distance, prominence=np.std(ecg_f)*0.2)
    if len(peaks) < 2:
        return hr_series
    # compute RR intervals and fill
    times = pd.to_datetime(df[time_column]).astype('int64') / 1e9  # seconds
    rr_s = np.diff(times[peaks])
    hr_beats = 60.0 / rr_s  # bpm between successive peaks
    # assign HR value between peaks
    for i in range(len(peaks)-1):
        start = peaks[i]
        end = peaks[i+1]
        hr_series.iloc[start:end] = hr_beats[i]
    # for tail, propagate last value
    hr_series.iloc[peaks[-1]:] = hr_series.iloc[peaks[-2]] if len(peaks)>1 else hr_series.iloc[peaks[-1]]
    # interpolate nan
    hr_series = hr_series.interpolate(method='linear', limit_direction='both')
    return hr_series

# main loop
all_cleaned = []
for filename in csv_files:
    print("\nProcessing:", filename)
    df = pd.read_csv(os.path.join(input_dir, filename))
    # parse time
    if 'time' not in df.columns:
        print("No time column, skipping", filename)
        continue
    df = parse_time_column(df, 'time')
    fs = estimate_fs(df, 'time')
    print("Estimated fs (Hz):", fs)

    # rename columns if present
    rename_map = {
        'temp_1': 'tempC',
        'a_x': 'accX', 'a_y': 'accY', 'a_z': 'accZ',
        'g_x': 'gyroX', 'g_y': 'gyroY', 'g_z': 'gyroZ',
    }
    df.rename(columns=rename_map, inplace=True)

    # ensure numeric where possible
    for c in df.columns:
        if c not in ['time']:
            try:
                df[c] = pd.to_numeric(df[c], errors='coerce')
            except:
                pass

    # normalize acc
    df, acc_src = detect_and_normalize_acc(df)
    print("Acc normalization:", acc_src)
    # normalize gyro
    df, gyro_src = detect_and_normalize_gyro(df)
    print("Gyro normalization:", gyro_src)

    # compute angles if needed
    if all(c in df.columns for c in ['accX','accY','accZ','gyroX','gyroY','gyroZ']):
        df = compute_angles_from_imu(df)
        print("Angles computed")

    # compute HR if missing and ecg present
    if 'hr' not in df.columns or df['hr'].isna().all():
        if 'ecg' in df.columns:
            hr_series = calculate_hr_from_ecg(df, ecg_column='ecg', time_column='time', fs=fs)
            df['hr'] = hr_series
            print("HR computed from ECG")
        else:
            df['hr'] = np.nan

    # keep only desired columns if exist
    existing_cols = [c for c in desired_columns if c in df.columns]
    df_out = df[existing_cols].copy()

    # fill/forward-backward any remaining NA in hr/time etc
    if 'hr' in df_out.columns:
        df_out['hr'] = df_out['hr'].interpolate(method='linear', limit_direction='both')

    # save cleaned
    out_path = os.path.join(output_dir, 'cleaned_' + filename)
    df_out.to_csv(out_path, index=False)
    print("Saved cleaned file:", out_path)
    all_cleaned.append(df_out)

# combine all
if len(all_cleaned) > 0:
    combined_df = pd.concat(all_cleaned, ignore_index=True)
    combined_df.to_csv('./combined_data.csv', index=False)
    print("Saved combined_data.csv")
else:
    print("No cleaned files to combine")
