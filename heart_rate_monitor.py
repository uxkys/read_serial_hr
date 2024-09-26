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
    nyq = 0.5 * fs  # ナイキスト周波数を計算
    low = lowcut / nyq  # 低周波数の正規化
    high = highcut / nyq  # 高周波数の正規化
    b, a = butter(order, [low, high], btype='band')  # バンドパスフィルタの係数を計算
    y = filtfilt(b, a, data)  # フィルタをデータに適用
    return y

# シリアル通信の設定
serial_port = '/dev/tty.usbmodem1101'  # 適切なポートを指定（環境に合わせて変更）
baud_rate = 115200  # 通信速度を設定
ser = serial.Serial(serial_port, baud_rate)

data = []  # 受信データを格納するリスト
fs = 200  # サンプリング周波数（Hz）

# 保存用のリスト
all_hr = []
all_rmssd = []
all_sdnn_rmssd = []
a_values = []

# CSVファイルを開く
with open('hr_data.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Timestamp', 'RRI (ms)', 'HR (bpm)'])  # ヘッダーを書き込む

    try:
        while True:
            line = ser.readline()  # シリアルポートから1行のデータを読み込む
            try:
                value = int(line.decode().strip())  # データを整数に変換
                data.append(value)  # データをリストに追加
            except ValueError:
                continue  # 整数に変換できない場合は無視して次に進む

            if len(data) >= fs * 10:  # データが10秒分以上蓄積されたら
                filtered_data = apply_bandpass_filter(np.array(data), lowcut=0.5, highcut=30, fs=fs)  # フィルタを適用
                r_peaks, _ = find_peaks(filtered_data, height=np.max(filtered_data) * 0.5, distance=fs * 0.6)  # Rピークを検出

                # RRIを計算（ms単位）
                rri = np.diff(r_peaks) / fs * 1000  # RRIを計算
                valid_rri = rri[(rri > 300) & (rri < 2000)]  # 有効なRRIのみを使用

                if len(valid_rri) > 1:
                    hr = 60000 / np.mean(valid_rri)  # 平均HRを計算
                    rmssd = np.sqrt(np.mean(np.diff(valid_rri) ** 2))  # RMSSDを計算
                    sdnn = np.std(valid_rri)  # SDNNを計算
                    sdnn_rmssd = sdnn / rmssd if rmssd != 0 else 0  # SDNN/RMSSDを計算

                    # 各値をリストに追加
                    all_hr.append(hr)
                    all_rmssd.append(rmssd)
                    all_sdnn_rmssd.append(sdnn_rmssd)
                    a_values.append(sdnn_rmssd)

                    # 最新の10秒間のA値を計算
                    a_last = sdnn_rmssd
                    # 直近5件のAの平均を計算
                    b_last = np.mean(a_values[-5:])
                    # 直近20件のAの平均を計算
                    c_last = np.mean(a_values[-20:])
                    # これまでの全SDNN/RMSSDの平均を計算
                    overall_mean_sdnn_rmssd = np.mean(all_sdnn_rmssd)

                    # 出力（改行せずに1行で出力）
                    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                    print(f"Ts, {timestamp}, Mean_HR, {np.mean(all_hr):.2f}, Mean_10s_HR, {hr:.2f}, Mean_10s_RRI, {np.mean(valid_rri):.3f}, "
                          f"Mean_10s_RMSSD, {rmssd:.3f}, A(SDNN/RMSSD), {a_last:.3f}, B(5_Avg_A), {b_last:.3f}, C(20_Avg_A), {c_last:.3f}, "
                          f"Overall_Mean_SDNN/RMSSD, {overall_mean_sdnn_rmssd:.3f}, A-Overall_Mean, {a_last - overall_mean_sdnn_rmssd:.3f}, "
                          f"A/Overall_Mean, {a_last / overall_mean_sdnn_rmssd:.3f}, B/Overall_Mean, {b_last / overall_mean_sdnn_rmssd:.3f}, "
                          f"C/Overall_Mean, {c_last / overall_mean_sdnn_rmssd:.3f}")

                    # CSVに書き込み
                    writer.writerow([timestamp, f"{np.mean(valid_rri):.3f}", f"{np.mean(all_hr):.2f}"])

                # データバッファをリセットして次のセグメントを処理
                data = []

    except KeyboardInterrupt:
        ser.close()  # シリアル通信を終了
