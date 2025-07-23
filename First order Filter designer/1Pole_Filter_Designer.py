import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons, Button
from scipy.signal import freqz, lfilter
import sounddevice as sd

# --- Globals ---
sample_rates = [44100, 48000]
current_fs = [48000]
stream = None

# Audio Stream Settings
BLOCK_SIZE  = pow(2, 14)    # 2^6(64) | 2*8(256) 
DATA_TYPE   ='int32'        # float32 | int16    | int32 | uint8 | float64
LATENCY     ='high'         # low     | high     | exclusive

# Slider limits for filter
FREQ_LOW            = 20
FREQ_HIGH           = 20000
LOW_BAND_FREQ_HIGH  = 10000
HIGH_BAND_FREQ_LOW  = 100

block_time = BLOCK_SIZE / current_fs[0]
block_ms = block_time * 1000  # convert to milliseconds
print(f"Block size:  {BLOCK_SIZE} samples")
print(f"Sample rate: {current_fs[0]} Hz")
print(f"Audio block: {block_ms:.2f} ms")

# --- Fixed-point Q8.24 ---
def q24_to_float(q): return q / (1 << 24)
def float_to_q24(f): return int(round(f * (1 << 24)))
def fc_to_q24(fc, fs=None):
    fs = fs or current_fs[0]
    return float_to_q24(1 - np.exp(-2 * np.pi * fc / fs))

def one_pole_iir_q24(a_q24, type='low'):
    a = q24_to_float(a_q24)
    if type == 'low':
        b = [a]
        a = [1, -(1 - a)]
    elif type == 'high':
        b = [1 - a, -(1 - a)]
        a = [1, -(1 - a)]
    else:
        raise ValueError("Invalid type")
    return b, a

# --- Filter coefficient generator ---
def get_filter_coeffs():
    fs = current_fs[0]
    mode = type_selector.value_selected

    if mode in ['LPF', 'HPF']:
        fc  = 10 ** fc_single_slider.val
        a_q24 = fc_to_q24(fc, fs)
        filter_type = 'low' if mode == 'LPF' else 'high'  # ← FIXED
        return one_pole_iir_q24(a_q24, filter_type)

    elif mode == 'BPF':
        fc_low  = 10 ** fc_low_slider.val
        fc_high  = 10 ** fc_high_slider.val
        a_low = fc_to_q24(fc_low, fs)
        a_high = fc_to_q24(fc_high, fs)
        b1, a1 = one_pole_iir_q24(a_low, 'high')
        b2, a2 = one_pole_iir_q24(a_high, 'low')
        return [b1, b2], [a1, a2]

    elif mode == 'BSF':
        fc_low  = 10 ** fc_low_slider.val
        fc_high  = 10 ** fc_high_slider.val
        a_low = fc_to_q24(fc_low, fs)
        a_high = fc_to_q24(fc_high, fs)
        b_lp, a_lp = one_pole_iir_q24(a_low, 'low')
        b_hp, a_hp = one_pole_iir_q24(a_high, 'high')
        return [b_lp, b_hp], [a_lp, a_hp]

# --- Audio callback ---
def audio_callback(indata, outdata, frames, time, status):
    if status: print(status)
    mode = type_selector.value_selected

    if mode in ['LPF', 'HPF']:
        b, a = get_filter_coeffs()
        for ch in range(2):
            outdata[:, ch] = lfilter(b, a, indata[:, ch])

    elif mode == 'BPF':
        b1, b2 = get_filter_coeffs()[0]
        a1, a2 = get_filter_coeffs()[1]
        for ch in range(2):
            x = lfilter(b1, a1, indata[:, ch])
            outdata[:, ch] = lfilter(b2, a2, x)

    elif mode == 'BSF':
        b_lp, b_hp = get_filter_coeffs()[0]
        a_lp, a_hp = get_filter_coeffs()[1]
        for ch in range(2):
            y_lp = lfilter(b_lp, a_lp, indata[:, ch])
            y_hp = lfilter(b_hp, a_hp, indata[:, ch])
            outdata[:, ch] = y_lp + y_hp

# --- Plot update ---
def update_plot(val):
    mode = type_selector.value_selected
    fs = current_fs[0]

    if mode in ['LPF', 'HPF']:
        fc  = 10 ** fc_single_slider.val
        a_q24 = fc_to_q24(fc, fs)
        coeff_texts["Fc"].set_text(f"Q24 = 0x{a_q24:08X}")

        filter_type = 'low' if mode == 'LPF' else 'high'
        b, a = one_pole_iir_q24(a_q24, filter_type)
        _, h = freqz(b, a, worN=frequencies, fs=fs)

    elif mode == 'BPF':
        fc_low  = 10 ** fc_low_slider.val
        fc_high  = 10 ** fc_high_slider.val
        a_low = fc_to_q24(fc_low, fs)
        a_high = fc_to_q24(fc_high, fs)
        coeff_texts["FcLow"].set_text(f"Q24 = 0x{a_low:08X}")
        coeff_texts["FcHigh"].set_text(f"Q24 = 0x{a_high:08X}")

        b1, a1 = one_pole_iir_q24(a_low, 'high')
        b2, a2 = one_pole_iir_q24(a_high, 'low')
        _, h1 = freqz(b1, a1, worN=frequencies, fs=fs)
        _, h2 = freqz(b2, a2, worN=frequencies, fs=fs)
        h = h1 * h2

    elif mode == 'BSF':
        fc_low  = 10 ** fc_low_slider.val
        fc_high  = 10 ** fc_high_slider.val
        a_low = fc_to_q24(fc_low, fs)
        a_high = fc_to_q24(fc_high, fs)
        coeff_texts["FcLow"].set_text(f"Q24 = 0x{a_low:08X}")
        coeff_texts["FcHigh"].set_text(f"Q24 = 0x{a_high:08X}")

        b_lp, a_lp = one_pole_iir_q24(a_low, 'low')
        b_hp, a_hp = one_pole_iir_q24(a_high, 'high')
        _, h_lp = freqz(b_lp, a_lp, worN=frequencies, fs=fs)
        _, h_hp = freqz(b_hp, a_hp, worN=frequencies, fs=fs)
        h = h_lp + h_hp

    response_line.set_ydata(20 * np.log10(np.maximum(np.abs(h), 1e-9)))
    fig.canvas.draw_idle()

# --- Sample rate change ---
def update_sample_rate(label):
    global stream
    current_fs[0] = int(label)
    if stream:
        stream.stop()
        stream.close()
    update_plot(None)

# --- Start audio stream ---
def start_audio(event):
    global stream
    if stream:
        stream.stop()
        stream.close()
    input_idx = int(input_selector.value_selected.split(':')[0])
    output_idx = int(output_selector.value_selected.split(':')[0])
    stream = sd.Stream(
        samplerate=current_fs[0],
        blocksize=BLOCK_SIZE,
        dtype=DATA_TYPE,
        channels=2,
        callback=audio_callback,
        latency=LATENCY,
        device=(input_idx, output_idx)
    )
    stream.start()

# --- Slider labels ---
def update_slider_labels():
    fc_single_slider.valtext.set_text(f"{10 ** fc_single_slider.val:.0f} Hz")
    fc_low_slider.valtext.set_text(f"{10 ** fc_low_slider.val:.0f} Hz")
    fc_high_slider.valtext.set_text(f"{10 ** fc_high_slider.val:.0f} Hz")

# --- Setup plot ---
fig, ax = plt.subplots(figsize=(10, 5))
plt.subplots_adjust(left=0.08, right=0.78, top=0.95, bottom=0.425)
frequencies = np.linspace(10, 24000, 2048)
response_line, = ax.semilogx(frequencies, np.zeros_like(frequencies))
ax.set_ylim(-40, 10)
ax.set_xlim(20, 20000)
ax.grid(True)
ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("Gain (dB)")
ax.set_title("1-Pole Filter Frequency Response")

# --- Sliders ---
fc_single_ax = plt.axes([0.15, 0.28, 0.5, 0.03])
fc_low_ax    = plt.axes([0.15, 0.24, 0.5, 0.03])  # was 0.25
fc_high_ax   = plt.axes([0.15, 0.20, 0.5, 0.03])

fc_single_slider = Slider(
    fc_single_ax, 'Cutoff (Hz)', 
    np.log10(FREQ_LOW), np.log10(FREQ_HIGH), 
    valinit=np.log10(1000), 
    valfmt='%1.0f Hz'
)

fc_low_slider = Slider(
    fc_low_ax, 'Low Fc (HPF)', 
    np.log10(FREQ_LOW), np.log10(LOW_BAND_FREQ_HIGH), 
    valinit=np.log10(300), 
    valfmt='%1.0f Hz'
)

fc_high_slider = Slider(
    fc_high_ax, 'High Fc (LPF)', 
    np.log10(HIGH_BAND_FREQ_LOW), np.log10(FREQ_HIGH), 
    valinit=np.log10(3000), 
    valfmt='%1.0f Hz'
)

# --- Coefficient labels ---
coeff_texts = {}

def add_q24_label(name, y):
    ax = plt.axes([0.75, y, 0.1, 0.03])  # left of slider
    ax.axis("off")
    coeff_texts[name] = ax.text(0, 0.5, "", fontsize=9, va='center')

# Y coordinates must match slider layout
add_q24_label("Fc",      0.28)
add_q24_label("FcLow",   0.24)
add_q24_label("FcHigh",  0.20)

# --- Filter type selector ---
type_ax = plt.axes([0.8, 0.4, 0.15, 0.15])
type_selector = RadioButtons(type_ax, ['LPF', 'HPF', 'BPF', 'BSF'], active=0)
for label in type_selector.labels:
    label.set_fontsize(8)

def toggle_sliders(label):
    if label in ['LPF', 'HPF']:
        fc_single_slider.ax.set_visible(True)
        fc_low_slider.ax.set_visible(False)
        fc_high_slider.ax.set_visible(False)
    else:
        fc_single_slider.ax.set_visible(False)
        fc_low_slider.ax.set_visible(True)
        fc_high_slider.ax.set_visible(True)
    update_plot(None)

type_selector.on_clicked(toggle_sliders)

# --- Device selection ---
wasapi_index = next(i for i, api in enumerate(sd.query_hostapis()) if api['name'].lower() == 'windows wasapi')
devices = sd.query_devices()
wasapi_devices = [(i, d['name'], d['max_input_channels'], d['max_output_channels'])
                  for i, d in enumerate(devices) if d['hostapi'] == wasapi_index]
input_devices = [(i, name) for i, name, ch_in, _ in wasapi_devices if ch_in > 0]
output_devices = [(i, name) for i, name, _, ch_out in wasapi_devices if ch_out > 0]

input_ax = plt.axes([0.1, 0.1, 0.35, 0.08])
output_ax = plt.axes([0.1, 0.01, 0.35, 0.08])
rate_ax = plt.axes([0.5, 0.01, 0.1, 0.05])
button_ax = plt.axes([0.63, 0.01, 0.2, 0.05])

input_selector = RadioButtons(input_ax, [f"{i}: {name}" for i, name in input_devices], active=0)
output_selector = RadioButtons(output_ax, [f"{i}: {name}" for i, name in output_devices], active=0)
rate_selector = RadioButtons(rate_ax, [str(r) for r in sample_rates], active=sample_rates.index(current_fs[0]))
rate_selector.on_clicked(update_sample_rate)
for selector in [input_selector, output_selector, rate_selector]:
    for label in selector.labels:
        label.set_fontsize(8)

start_button = Button(button_ax, 'Start Audio')
start_button.on_clicked(start_audio)

update_slider_labels()

# --- Hook up sliders ---
fc_single_slider.on_changed(lambda val: (update_plot(val), update_slider_labels()))
fc_low_slider.on_changed(lambda val: (update_plot(val), update_slider_labels()))
fc_high_slider.on_changed(lambda val: (update_plot(val), update_slider_labels()))

toggle_sliders('LPF')
update_plot(None)
plt.show()
