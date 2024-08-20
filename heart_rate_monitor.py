import serial
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks
import time

# バンドパスフィルタを適用する関数
def apply_bandpass_filter(data, lowcut=0.5, highcut=30, fs=200, order=5):
    nyq = 0.5 * fs  # ナイキスト周波数
    low = lowcut / nyq  # 通過帯域の下限
    high = highcut / nyq  # 通過帯域の上限
    b, a = butter(order, [low, high], btype='band')  # フィルタ係数の計算
    y = filtfilt(b, a, data)  # データにフィルタを適用
    return y

# 心拍変動指標を計算する関数
def compute_heart_rate_variability(r_peaks, fs=200):
    if len(r_peaks) > 1:  # R波のピークが2つ以上検出された場合にのみ計算を行う
        rr_intervals = np.diff(r_peaks) / fs * 1000  # RRIをミリ秒単位で計算
        rmssd = np.sqrt(np.mean(np.square(np.diff(rr_intervals))))  # RMSSDを計算
        sdnn = np.std(rr_intervals)  # SDNNを計算
        sdnn_rmssd_ratio = sdnn / rmssd if rmssd != 0 else np.nan  # SDNN/RMSSD比を計算
        return rmssd, sdnn, sdnn_rmssd_ratio
    else:
        return np.nan, np.nan, np.nan  # R波が十分検出されない場合はNaNを割り当て

# シリアルポートの設定
serial_port = '/dev/tty.usbmodem11301'
baud_rate = 115200
ser = serial.Serial(serial_port, baud_rate)

# 出力ファイルの準備
output_file = open("hrv_data.csv", "w")  # ファイルを開く
output_file.write("Timestamp,RMSSD,SDNN,SDNN_RMSSD\n")  # ヘッダーを書き込む

data = []
try:
    while True:
        line = ser.readline()  # シリアルポートからデータを読み込む
        try:
            value = int(line.decode().strip())  # 受信データを整数に変換
            data.append(value)  # データをリストに追加
        except ValueError:
            continue  # 変換できないデータは無視

        # 一定のデータ量に達したら処理を行う
        if len(data) >= 200 * 10:  # 例えば10秒分のデータを溜める
            filtered_data = apply_bandpass_filter(np.array(data))  # データにバンドパスフィルタを適用
            r_peaks, _ = find_peaks(filtered_data, height=np.max(filtered_data)*0.5)  # R波のピークを検出
            rmssd, sdnn, sdnn_rmssd_ratio = compute_heart_rate_variability(r_peaks)  # 心拍変動指標を計算
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")  # 現在の時刻を取得
            output_file.write(f"{timestamp},{rmssd},{sdnn},{sdnn_rmssd_ratio}\n")  # ファイルに書き込み
            print("RMSSD:", rmssd, "SDNN:", sdnn, "SDNN/RMSSD:", sdnn_rmssd_ratio)  # 結果を表示
            data = []  # データバッファをクリア

except KeyboardInterrupt:
    output_file.close()  # ファイルを閉じる
    ser.close()  # シリアルポートを閉じる
