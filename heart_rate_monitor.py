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

# --- pyHRVライブラリをインポート ---
from pyhrv.time_domain import time_domain

# --------------------------------------------------
# バンドパスフィルタを適用する関数
# --------------------------------------------------
def apply_bandpass_filter(data, lowcut=0.5, highcut=30, fs=200, order=5):
    nyq = 0.5 * fs  # ナイキスト周波数
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, data)
    return y

# --------------------------------------------------
# A(SDNN/RMSSD)から覚醒度(0-100)を計算する関数
# --------------------------------------------------
def compute_arousal_value(a_current, meanA, stdA):
    lower_bound = meanA - stdA  # 下限
    upper_bound = meanA + stdA  # 上限
    if upper_bound - lower_bound == 0:
        return 50.0  # 分母0対策
    arousal = (a_current - lower_bound) / (upper_bound - lower_bound) * 100.0
    arousal = max(0.0, min(100.0, arousal))  # 0-100にクリップ
    return arousal

# --------------------------------------------------
# シリアル通信の設定
# --------------------------------------------------
serial_port = '/dev/cu.usbmodem11301'
baud_rate = 115200
ser = serial.Serial(serial_port, baud_rate)

# --------------------------------------------------
# パラメータの初期設定
# --------------------------------------------------
data = []
fs = 200  # サンプリング周波数（Hz）

all_hr = []
all_sdnn_rmssd = []
a_values = []  # A(SDNN/RMSSD) の履歴

csv_filename = 'hr_data.csv'

# -----------------------------
# ベースライン計測用の変数
# -----------------------------
baseline_values = []
measure_baseline = True
baseline_duration_sec = 300  # 例：5分(300秒)
baseline_start_time = time.time()

meanA_baseline = None
stdA_baseline = None

# カウントダウン表示のための変数
previous_remaining_time = baseline_duration_sec  # 前回表示した残り時間（秒）

# --------------------------------------------------
# CSVファイルを開いてメイン処理開始
# --------------------------------------------------
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

            # 10秒分たまったら解析する
            if len(data) >= fs * 10:
                # バンドパスフィルタ
                filtered_data = apply_bandpass_filter(
                    np.array(data), lowcut=0.5, highcut=30, fs=fs
                )
                # Rピーク検出
                r_peaks, _ = find_peaks(
                    filtered_data,
                    height=np.max(filtered_data) * 0.5,
                    distance=fs * 0.6
                )

                # RRI計算（ms単位）
                rri = np.diff(r_peaks) / fs * 1000
                valid_rri = rri[(rri > 300) & (rri < 2000)]

                if len(valid_rri) > 1:
                    # ------------------------------------------
                    # pyHRVを使ってHRVの各指標を算出
                    # ------------------------------------------
                    result = time_domain(valid_rri, rpeaks=None, show=False)

                    # 返されるキーを get() で取得し、無い場合はNone
                    hr    = result.get('mean_hr', None)
                    rmssd = result.get('rmssd', None)
                    sdnn  = result.get('sdnn', None)

                    # もし mean_hr が取れなければ、valid_rri の平均から近似的に計算
                    if hr is None:
                        mean_rri = np.mean(valid_rri)  # ms
                        if mean_rri > 0:
                            hr = 60000.0 / mean_rri  # (bpm)
                        else:
                            hr = 0.0

                    # rmssd, sdnn も None の場合があるので、0で代替
                    if rmssd is None:
                        rmssd = 0.0
                    if sdnn is None:
                        sdnn = 0.0

                    # SDNN/RMSSD (0割り対策)
                    if rmssd == 0.0:
                        sdnn_rmssd = 0.0
                    else:
                        sdnn_rmssd = sdnn / rmssd

                    # ---------------------------------------------
                    # A(SDNN/RMSSD)としてリストに追加・ベースライン計測
                    # ---------------------------------------------
                    a_values.append(sdnn_rmssd)
                    all_hr.append(hr)
                    all_sdnn_rmssd.append(sdnn_rmssd)

                    # ベースライン測定中（300秒）
                    if measure_baseline:
                        baseline_values.append(sdnn_rmssd)
                        elapsed_time = time.time() - baseline_start_time
                        remaining_time = int(baseline_duration_sec - elapsed_time)

                        # 1秒ごとに残り時間が減ったタイミングで表示
                        if remaining_time < 0:
                            remaining_time = 0
                        if remaining_time != previous_remaining_time:
                            print(f"[Baseline] Remaining: {remaining_time} sec")
                            previous_remaining_time = remaining_time

                        # ベースライン計測終了判定
                        if elapsed_time >= baseline_duration_sec:
                            meanA_baseline = np.mean(baseline_values)
                            stdA_baseline = np.std(baseline_values)
                            measure_baseline = False
                            print("=== Baseline measurement finished ===")
                            print(f"Baseline meanA={meanA_baseline:.3f}, stdA={stdA_baseline:.3f}")

                    # ベースライン終了後にarousal計算
                    if (meanA_baseline is not None) and (stdA_baseline is not None):
                        current_arousal = compute_arousal_value(
                            sdnn_rmssd, meanA_baseline, stdA_baseline
                        )
                    else:
                        current_arousal = 50.0

                    # --------------------------------------------------
                    # ターミナル出力
                    # --------------------------------------------------
                    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                    mean_10s_rri = np.mean(valid_rri)
                    print(
                        f"Ts, {timestamp}, "
                        f"Mean_HR(All), {np.mean(all_hr):.2f}, "
                        f"Mean_10s_HR, {hr:.2f}, "
                        f"Mean_10s_RRI, {mean_10s_rri:.3f}, "
                        f"Mean_10s_RMSSD, {rmssd:.3f}, "
                        f"A(SDNN/RMSSD), {sdnn_rmssd:.3f}, "
                        f"Arousal(0-100), {current_arousal:.1f}"
                    )

                    # --------------------------------------------------
                    # CSVに書き込み
                    # --------------------------------------------------
                    writer.writerow([
                        timestamp,
                        f"{mean_10s_rri:.3f}",
                        f"{np.mean(all_hr):.2f}",
                        f"{current_arousal:.2f}"
                    ])

                # データバッファをリセット
                data = []

    except KeyboardInterrupt:
        ser.close()
        print("Serial port closed. End.")
