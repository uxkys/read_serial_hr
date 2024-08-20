import serial
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks
import time

def apply_bandpass_filter(data, lowcut=0.5, highcut=30, fs=200, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, data)
    return y

def compute_heart_rate_variability(r_peaks, fs=200):
    if len(r_peaks) > 1:
        rr_intervals = np.diff(r_peaks) / fs * 1000
        rmssd = np.sqrt(np.mean(np.square(np.diff(rr_intervals))))
        sdnn = np.std(rr_intervals)
        sdnn_rmssd_ratio = sdnn / rmssd if rmssd != 0 else np.nan
        return rmssd, sdnn, sdnn_rmssd_ratio
    else:
        return np.nan, np.nan, np.nan

def moving_average(data, n=5):
    return np.convolve(data, np.ones(n)/n, mode='valid')

serial_port = '/dev/tty.usbmodem11301'
baud_rate = 115200
ser = serial.Serial(serial_port, baud_rate)

output_file = open("hrv_data.csv", "w")
output_file.write("Timestamp,RMSSD,SDNN,SDNN_RMSSD,Short_MA,Long_MA,MA_Difference\n")

data = []
rmssd_values = []
sdnn_values = []

try:
    while True:
        line = ser.readline()
        try:
            value = int(line.decode().strip())
            data.append(value)
        except ValueError:
            continue

        if len(data) >= 200 * 10:
            filtered_data = apply_bandpass_filter(np.array(data))
            r_peaks, _ = find_peaks(filtered_data, height=np.max(filtered_data)*0.5)
            rmssd, sdnn, sdnn_rmssd_ratio = compute_heart_rate_variability(r_peaks)
            rmssd_values.append(rmssd)
            sdnn_values.append(sdnn)

            if len(rmssd_values) >= 20:
                short_ma_rmssd = moving_average(rmssd_values[-20:], 5)[-1]
                long_ma_rmssd = moving_average(rmssd_values, 20)[-1]
                ma_difference_rmssd = short_ma_rmssd - long_ma_rmssd

                timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                output_line = f"{timestamp},{rmssd},{sdnn},{sdnn_rmssd_ratio},{short_ma_rmssd},{long_ma_rmssd},{ma_difference_rmssd}\n"
                output_file.write(output_line)
                print(f"RMSSD: {rmssd}, SDNN: {sdnn}, SDNN/RMSSD: {sdnn_rmssd_ratio}, Short MA: {short_ma_rmssd}, Long MA: {long_ma_rmssd}, MA Difference: {ma_difference_rmssd}")
                data = []  # Reset data buffer for next segment

except KeyboardInterrupt:
    output_file.close()
    ser.close()
