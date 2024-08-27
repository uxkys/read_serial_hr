import serial  # シリアル通信を扱うためのライブラリをインポート
import numpy as np  # 数値計算のためのライブラリをインポート
from scipy.signal import butter, filtfilt, find_peaks  # 信号処理に必要な関数をインポート
import time  # 時間に関連する関数を扱うためのライブラリをインポート

# ▼ターミナルコマンド▼
# cd Box\ Sync/書類/github/read-serial
# python3 heart_rate_monitor.py
# ctrl + C と入力するとpythonから抜けられます
# ▲ターミナルコマンド▲

# バンドパスフィルタを適用する関数
def apply_bandpass_filter(data, lowcut=0.5, highcut=30, fs=200, order=5):
    nyq = 0.5 * fs  # ナイキスト周波数を計算
    low = lowcut / nyq  # 低周波数の正規化
    high = highcut / nyq  # 高周波数の正規化
    b, a = butter(order, [low, high], btype='band')  # バンドパスフィルタの係数を計算
    y = filtfilt(b, a, data)  # フィルタをデータに適用
    return y

# 心拍変動（HRV）を計算する関数
def compute_heart_rate_variability(r_peaks, fs=200):
    if len(r_peaks) > 1:  # ピークが2つ以上ある場合
        rr_intervals = np.diff(r_peaks) / fs * 1000  # R-R間隔（ms単位）を計算
        rmssd = np.sqrt(np.mean(np.square(np.diff(rr_intervals))))  # RMSSDを計算
        sdnn = np.std(rr_intervals)  # SDNNを計算
        sdnn_rmssd_ratio = sdnn / rmssd if rmssd != 0 else np.nan  # SDNN/RMSSD比を計算
        return rmssd, sdnn, sdnn_rmssd_ratio  # 結果を返す
    else:
        return np.nan, np.nan, np.nan  # ピークが不足している場合はNaNを返す

# 移動平均を計算する関数
def moving_average(data, n=5):
    if len(data) < n:
        return np.array([np.nan])  # データが不足している場合はNaNを返す
    return np.convolve(data, np.ones(n)/n, mode='valid')  # 移動平均を計算

serial_port = '/dev/tty.usbmodem11301'  # 使用するシリアルポートを指定
baud_rate = 115200  # 通信速度を設定
ser = serial.Serial(serial_port, baud_rate)  # シリアル通信を開始

# データを保存するファイルを開く
output_file = open("hrv_data.csv", "w")
# ファイルのヘッダーを書き込む
output_file.write("Timestamp,RMSSD,SDNN,SDNN_RMSSD,Short_MA,Long_MA,Cumulative_Avg_SDNN_RMSSD,MA_Difference,Cumulative_Avg_Long_MA_Difference,Cumulative_Avg_Short_MA_Difference\n")

data = []  # 受信データを格納するリスト
rmssd_values = []  # RMSSDの計算結果を格納するリスト
sdnn_values = []  # SDNNの計算結果を格納するリスト
sdnn_rmssd_values = []  # SDNN/RMSSDの計算結果を格納するリスト
cumulative_sum_sdnn_rmssd = 0  # SDNN/RMSSDの累積和
count = 0  # 計算されたSDNN/RMSSDの個数

try:
    while True:  # 無限ループでデータを処理
        line = ser.readline()  # シリアルポートから1行のデータを読み込む
        try:
            value = int(line.decode().strip())  # データを整数に変換
            data.append(value)  # データをリストに追加
        except ValueError:
            continue  # 整数に変換できない場合は無視して次に進む

        if len(data) >= 200 * 10:  # データが10秒分以上蓄積されたら
            filtered_data = apply_bandpass_filter(np.array(data))  # フィルタを適用
            r_peaks, _ = find_peaks(filtered_data, height=np.max(filtered_data)*0.5)  # Rピークを検出
            rmssd, sdnn, sdnn_rmssd_ratio = compute_heart_rate_variability(r_peaks)  # HRVを計算

            if not np.isnan(rmssd):
                rmssd_values.append(rmssd)
            if not np.isnan(sdnn):
                sdnn_values.append(sdnn)
            if not np.isnan(sdnn_rmssd_ratio):
                sdnn_rmssd_values.append(sdnn_rmssd_ratio)
                cumulative_sum_sdnn_rmssd += sdnn_rmssd_ratio  # 累積和に追加
                count += 1  # カウントを1増やす
                cumulative_avg_sdnn_rmssd = cumulative_sum_sdnn_rmssd / count  # 累積平均を計算

                # 短期移動平均と長期移動平均を計算
                if len(sdnn_rmssd_values) >= 20:
                    valid_sdnn_rmssd_values = [v for v in sdnn_rmssd_values[-20:] if not np.isnan(v)]  # 直近20個のNaNでない値を取得
                    valid_long_sdnn_rmssd_values = [v for v in sdnn_rmssd_values if not np.isnan(v)]  # 全てのNaNでない値を取得

                    short_ma_sdnn_rmssd = moving_average(valid_sdnn_rmssd_values, 5)[-1] if len(valid_sdnn_rmssd_values) >= 5 else np.nan  # 短期移動平均を計算
                    long_ma_sdnn_rmssd = moving_average(valid_long_sdnn_rmssd_values, 20)[-1] if len(valid_long_sdnn_rmssd_values) >= 20 else np.nan  # 長期移動平均を計算
                    ma_difference_sdnn_rmssd = short_ma_sdnn_rmssd - long_ma_sdnn_rmssd if not np.isnan(short_ma_sdnn_rmssd) and not np.isnan(long_ma_sdnn_rmssd) else np.nan  # 短期移動平均と長期移動平均の差分を計算

                    # 累積平均と移動平均の差分を計算
                    cumulative_avg_long_ma_difference = cumulative_avg_sdnn_rmssd - long_ma_sdnn_rmssd if not np.isnan(long_ma_sdnn_rmssd) else np.nan
                    cumulative_avg_short_ma_difference = cumulative_avg_sdnn_rmssd - short_ma_sdnn_rmssd if not np.isnan(short_ma_sdnn_rmssd) else np.nan
                else:
                    short_ma_sdnn_rmssd = np.nan
                    long_ma_sdnn_rmssd = np.nan
                    ma_difference_sdnn_rmssd = np.nan
                    cumulative_avg_long_ma_difference = np.nan
                    cumulative_avg_short_ma_difference = np.nan

                timestamp = time.strftime("%Y-%m-%d %H:%M:%S")  # 現在のタイムスタンプを取得
                output_line = (
                    f"{timestamp},{rmssd:.3f},{sdnn:.3f},{sdnn_rmssd_ratio:.3f},{short_ma_sdnn_rmssd:.3f},{long_ma_sdnn_rmssd:.3f},"
                    f"{cumulative_avg_sdnn_rmssd:.3f},{ma_difference_sdnn_rmssd:.3f},{cumulative_avg_long_ma_difference:.3f},{cumulative_avg_short_ma_difference:.3f}\n"
                )
                output_file.write(output_line)  # データをファイルに書き込む
                
                # 出力を指定されたフォーマットで表示
                print(
                    f"RMSSD: {rmssd:.3f}, SDNN: {sdnn:.3f}, SDNN/RMSSD: {sdnn_rmssd_ratio:.3f}, SMAS/R: {short_ma_sdnn_rmssd:.3f}, "
                    f"LMAS/R: {long_ma_sdnn_rmssd:.3f}, AllAvgS/R: {cumulative_avg_sdnn_rmssd:.3f}, "
                    f"S-L: {ma_difference_sdnn_rmssd:.3f}, All-L: {cumulative_avg_long_ma_difference:.3f}, "
                    f"All-S: {cumulative_avg_short_ma_difference:.3f}"
                )
                
            data = []  # データバッファをリセットして次のセグメントを処理

except KeyboardInterrupt:  # プログラムが停止されたとき
    output_file.close()  # ファイルを閉じる
    ser.close()  # シリアル通信を終了
