# ▼ターミナルコマンド▼
# cd Box\ Sync/書類/github/read-serial
# python3 heart_rate_monitor.py
# ctrl + C と入力するとpythonから抜けられます
# ▲ターミナルコマンド▲

import serial
import numpy as np
import csv
import time
from scipy.signal import butter, filtfilt, find_peaks

# バンドパスフィルタを適用する関数
def apply_bandpass_filter(data, lowcut=0.5, highcut=30, fs=200, order=5):
    nyq = 0.5 * fs  # ナイキスト周波数
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, data)
    return y

# A(SDNN/RMSSD)から覚醒度(0-100)を計算する関数
# ベースライン平均(meanA)±標準偏差(stdA)を閾値とし、
# (meanA - stdA)→0, (meanA + stdA)→100 の範囲に線形マッピングする例
def compute_arousal_value(a_current, meanA, stdA):
    lower_bound = meanA - stdA  # 下限
    upper_bound = meanA + stdA  # 上限

    if upper_bound - lower_bound == 0:
        return 50.0  # 分母0対策

    arousal = (a_current - lower_bound) / (upper_bound - lower_bound) * 100.0
    arousal = max(0.0, min(100.0, arousal))  # 0-100にクリップ
    return arousal

# シリアル通信の設定
serial_port = '/dev/tty.usbmodem1101'
baud_rate = 115200
ser = serial.Serial(serial_port, baud_rate)

# パラメータの初期設定
data = []
fs = 200  # サンプリング周波数（Hz）

all_hr = []
all_rmssd = []
all_sdnn_rmssd = []
a_values = []  # A(SDNN/RMSSD) の履歴

csv_filename = 'hr_data.csv'

# ベースライン計測用の変数
baseline_values = []
measure_baseline = True
baseline_duration_sec = 300  # ★ ここを 5分(300秒)に変更 ★
baseline_start_time = time.time()

meanA_baseline = None
stdA_baseline = None

with open(csv_filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Timestamp', 'RRI (ms)', 'HR (bpm)', 'Arousal(0-100)'])

    try:
        while True:
            line = ser.readline()
            try:
                value = int(line.decode().strip())
                data.append(value)
            except ValueError:
                continue

            # 10秒たまるごとに解析
            if len(data) >= fs * 10:
                filtered_data = apply_bandpass_filter(
                    np.array(data), lowcut=0.5, highcut=30, fs=fs
                )
                r_peaks, _ = find_peaks(
                    filtered_data,
                    height=np.max(filtered_data) * 0.5,
                    distance=fs * 0.6
                )

                rri = np.diff(r_peaks) / fs * 1000
                valid_rri = rri[(rri > 300) & (rri < 2000)]

                if len(valid_rri) > 1:
                    hr = 60000.0 / np.mean(valid_rri)
                    rmssd = np.sqrt(np.mean(np.diff(valid_rri) ** 2))
                    sdnn = np.std(valid_rri)
                    sdnn_rmssd = sdnn / rmssd if rmssd != 0 else 0

                    a_values.append(sdnn_rmssd)
                    all_hr.append(hr)
                    all_rmssd.append(rmssd)
                    all_sdnn_rmssd.append(sdnn_rmssd)

                    # ベースライン計測(5分間)
                    if measure_baseline:
                        baseline_values.append(sdnn_rmssd)
                        elapsed_time = time.time() - baseline_start_time
                        if elapsed_time >= baseline_duration_sec:
                            meanA_baseline = np.mean(baseline_values)
                            stdA_baseline = np.std(baseline_values)
                            measure_baseline = False
                            print("=== Baseline measurement finished ===")
                            print(f"Baseline meanA={meanA_baseline:.3f}, stdA={stdA_baseline:.3f}")

                    # ベースライン確定後
                    if (meanA_baseline is not None) and (stdA_baseline is not None):
                        current_arousal = compute_arousal_value(
                            sdnn_rmssd, meanA_baseline, stdA_baseline
                        )
                    else:
                        current_arousal = 50.0

                    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                    print(
                        f"Ts, {timestamp}, "
                        f"Mean_HR, {np.mean(all_hr):.2f}, "
                        f"Mean_10s_HR, {hr:.2f}, "
                        f"Mean_10s_RRI, {np.mean(valid_rri):.3f}, "
                        f"Mean_10s_RMSSD, {rmssd:.3f}, "
                        f"A(SDNN/RMSSD), {sdnn_rmssd:.3f}, "
                        f"Arousal(0-100), {current_arousal:.1f}"
                    )

                    writer.writerow([
                        timestamp,
                        f"{np.mean(valid_rri):.3f}",
                        f"{np.mean(all_hr):.2f}",
                        f"{current_arousal:.2f}"
                    ])

                data = []

    except KeyboardInterrupt:
        ser.close()
        print("Serial port closed. End.")
