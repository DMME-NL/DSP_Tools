import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons, Button
from scipy.signal import freqz, lfilter
import sounddevice as sd
import math

# Audio Stream Settings
BLOCK_SIZE  = pow(2, 15)    # 2^6(64) | 2*8(256) 
DATA_TYPE   ='int32'        # float32 | int16    | int32 | uint8 | float64
LATENCY     ='high'         # low     | high     | exclusive

# --- Global Config ---
sample_rates = [44100, 48000]
current_fs = [48000]  # mutable container
PLOT_FS = 48000
no_points = 4096
stream = None

def print_stream_detail():
    block_time = BLOCK_SIZE / current_fs[0]
    block_ms = block_time * 1000  # convert to milliseconds
    print(f"Block size:  {BLOCK_SIZE} samples")
    print(f"Sample rate: {current_fs[0]} Hz")
    print(f"Audio block: {block_ms:.2f} ms")

print_stream_detail()

# --- Conversion Utilities ---
def q24_to_float(q): return q / (1 << 24)
def float_to_q24(f): return int(round(f * (1 << 24)))
def fc_to_q24(fc, fsample=None):
    fsample = fsample or current_fs[0]
    a = 1 - np.exp(-2 * np.pi * fc / fsample)
    return int(round(a * (1 << 24)))
def q24_to_fc(a_q24, fsample=None):
    fsample = fsample or current_fs[0]
    a = q24_to_float(a_q24)
    if a >= 1.0: return fsample / 2
    return -math.log(1 - a) * fsample / (2 * math.pi)

def one_pole_iir_q24(a_q24, type='low'):
    a = q24_to_float(a_q24)
    if type == 'low':
        b = [a]
        a = [1, -(1 - a)]
    elif type == 'high':
        b = [1 - a, -(1 - a)]
        a = [1, -(1 - a)]
    else:
        raise ValueError("type must be 'low' or 'high'")
    return b, a

def db(x): return 20 * np.log10(np.maximum(np.abs(x), 1e-10))

# --- Get coefficients and sliders ---
def get_filter_coeffs(fsample=None):
    fsample = fsample or current_fs[0]
    HPF_A   = fc_to_q24(hpf_slider.val, fsample)
    LOW_A   = fc_to_q24(low_slider.val, fsample)
    MID_A   = fc_to_q24(mid_slider.val, fsample)
    HIGH_A  = fc_to_q24(high_slider.val, fsample)
    LPF_A   = fc_to_q24(lpf_slider.val, fsample)

    low_gain  = bass_gain_slider.val
    mid_gain  = mid_gain_slider.val
    high_gain = treble_gain_slider.val

    return {
        'hpf': one_pole_iir_q24(HPF_A, 'high'),
        'low': one_pole_iir_q24(LOW_A, 'low'),
        'mid_lp': one_pole_iir_q24(MID_A, 'low'),
        'mid_hp': one_pole_iir_q24(MID_A, 'high'),
        'high': one_pole_iir_q24(HIGH_A, 'high'),
        'lpf': one_pole_iir_q24(LPF_A, 'low'),
        'gains': (low_gain, mid_gain, high_gain),
        'A': {
            'HPF': HPF_A, 'LOW': LOW_A, 'MID': MID_A,
            'HIGH': HIGH_A, 'LPF': LPF_A
        }
    }

def update_slider_labels():
    fs = PLOT_FS
    A_vals = get_filter_coeffs()['A']

    sliders = {
        "HPF": hpf_slider,
        "LOW": low_slider,
        "MID": mid_slider,
        "HIGH": high_slider,
        "LPF": lpf_slider,
    }

    for label, slider in sliders.items():
        fc_effective = q24_to_fc(A_vals[label], fs)

def update(val):
    # Use selected sample rate to plot the frequency response
    coeffs = get_filter_coeffs(fsample=current_fs[0])
    fs = current_fs[0]
    A_vals = coeffs['A']
    for key in A_vals:
        coeff_texts[key].set_text(f"Q24 = 0x{A_vals[key]:08X}")

    # Use fixed sample rate to plot the frequency response
    coeffs = get_filter_coeffs(PLOT_FS)
    b_hpf, a_hpf = coeffs['hpf']
    b_lpf, a_lpf = coeffs['lpf']
    b_low, a_low = coeffs['low']
    b_mid_lp, a_mid_lp = coeffs['mid_lp']
    b_mid_hp, a_mid_hp = coeffs['mid_hp']
    b_high, a_high = coeffs['high']
    low_gain, mid_gain, high_gain = coeffs['gains']

    update_slider_labels()

    _, h_hpf = freqz(b_hpf, a_hpf, worN=no_points, fs=current_fs[0])
    _, h_lpf = freqz(b_lpf, a_lpf, worN=no_points, fs=current_fs[0])
    _, h_low = freqz(b_low, a_low, worN=no_points, fs=current_fs[0])
    _, h_mid_lp = freqz(b_mid_lp, a_mid_lp, worN=no_points, fs=current_fs[0])
    _, h_mid_hp = freqz(b_mid_hp, a_mid_hp, worN=no_points, fs=current_fs[0])
    _, h_high = freqz(b_high, a_high, worN=no_points, fs=current_fs[0])

    h_pre = h_hpf * h_lpf
    h_mid = h_mid_lp * h_mid_hp
    h_total = (h_low * low_gain + h_mid * mid_gain + h_high * high_gain) * h_pre

    line_total.set_ydata(db(h_total))
    lines[0].set_ydata(db(h_low * low_gain))
    lines[1].set_ydata(db(h_mid * mid_gain))
    lines[2].set_ydata(db(h_high * high_gain))
    lines[3].set_ydata(db(h_hpf))
    lines[4].set_ydata(db(h_lpf))
    fig.canvas.draw_idle()

    line_total.set_xdata(frequencies)
    for line in lines:
        line.set_xdata(frequencies)

def update_sample_rate(label):
    global stream
    new_fs = int(label)
    current_fs[0] = new_fs

    if stream:
        stream.stop()
        stream.close()
        stream = None

    print_stream_detail()

    # Do NOT touch sliders — let user-specified frequencies remain!
    # Instead: just update Q24 values and re-render everything
    update(None)


def audio_callback(indata, outdata, frames, time, status):
    if status:
        print(status)
    coeffs = get_filter_coeffs()
    low_gain, mid_gain, high_gain = coeffs['gains']
    out_channels = []
    for ch in range(2):
        s = indata[:, ch]
        s = lfilter(*coeffs['hpf'], s)
        s = lfilter(*coeffs['lpf'], s)
        low  = lfilter(*coeffs['low'], s) * low_gain
        mid  = lfilter(*coeffs['mid_lp'], lfilter(*coeffs['mid_hp'], s)) * mid_gain
        high = lfilter(*coeffs['high'], s) * high_gain
        out_channels.append(low + mid + high)
    outdata[:] = np.stack(out_channels, axis=-1)

def start_audio(event):
    global stream
    input_idx = int(input_selector.value_selected.split(':')[0])
    output_idx = int(output_selector.value_selected.split(':')[0])
    stream = sd.Stream(
        samplerate=current_fs[0], 
        blocksize=BLOCK_SIZE, 
        dtype=DATA_TYPE, 
        latency=LATENCY,
        channels=2, 
        callback=audio_callback, 
        device=(input_idx, output_idx))
    stream.start()

# --- Setup Figure ---
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
plt.subplots_adjust(left=0.08, right=0.98, top=0.95, bottom=0.32, hspace=0.15)
frequencies = np.linspace(10, current_fs[0] // 2, no_points)
zeros = np.full_like(frequencies, -100)

line_total = ax1.semilogx(frequencies, zeros, label="Total Tone Stack", color='blue')[0]
lines = [ax2.semilogx(frequencies, zeros, '--', label=lbl)[0] for lbl in ["Low Shelf", "Mid Band", "High Shelf"]]
lines += [ax2.semilogx(frequencies, zeros, ':', label=lbl)[0] for lbl in ["HPF", "LPF"]]
ax1.set_title("Total Tone Stack")
ax2.set_title("Individual Tone Stack Bands")
ax2.set_xlabel("Frequency (Hz)")
ax1.set_ylabel("Magnitude (dB)")
ax2.set_ylabel("Magnitude (dB)")
ax1.grid(True, which='both')
ax2.grid(True, which='both')
ax1.set_ylim(-30, 12)
ax2.set_ylim(-60, 12)
ax2.set_xlim(20, 20000)
ax1.legend()
ax2.legend()

# --- Sliders between plots ---
slider_y_start = 0.22
slider_height = 0.02
slider_spacing = 0.025
slider_axs = {}
labels = ["Bass Gain", "Mid Gain", "Treble Gain", "HPF Fc", "Low Fc", "Mid Fc", "High Fc", "LPF Fc"]
for i, label in enumerate(labels):
    y_pos = slider_y_start - i * slider_spacing
    slider_axs[label] = plt.axes([0.1, y_pos, 0.3, slider_height])

bass_gain_slider   = Slider(slider_axs["Bass Gain"], "Bass Gain", 0.0, 2.0, valinit=1.0)
mid_gain_slider    = Slider(slider_axs["Mid Gain"], "Mid Gain", 0.0, 3.0, valinit=1.0)
treble_gain_slider = Slider(slider_axs["Treble Gain"], "Treble Gain", 0.0, 2.0, valinit=1.0)
hpf_slider  = Slider(slider_axs["HPF Fc"], "HPF Fc", 20, 150, valinit=90)
low_slider  = Slider(slider_axs["Low Fc"], "Low Fc", 50, 400, valinit=120)
mid_slider  = Slider(slider_axs["Mid Fc"], "Mid Fc", 300, 2500, valinit=600)
high_slider = Slider(slider_axs["High Fc"], "High Fc", 1000, 6000, valinit=3200)
lpf_slider  = Slider(slider_axs["LPF Fc"], "LPF Fc", 1000, 16000, valinit=6500)

coeff_texts = {}
def add_coeff_label(label, y):
    ax = plt.axes([0.45, y, 0.2, 0.02])
    ax.axis("off")
    coeff_texts[label] = ax.text(0, 0, "", fontsize=9, va='center')

for i, (lbl, _) in enumerate(zip(["HPF", "LOW", "MID", "HIGH", "LPF"], labels[3:])):
    y_pos = slider_y_start - (i + 2.5) * slider_spacing
    add_coeff_label(lbl, y_pos)

for s in [bass_gain_slider, mid_gain_slider, treble_gain_slider,
          hpf_slider, low_slider, mid_slider, high_slider, lpf_slider]:
    s.on_changed(update)

# --- Audio Device + Start UI (small, LEFT side) ---
wasapi_index = next(i for i, api in enumerate(sd.query_hostapis()) if api['name'].lower() == 'windows wasapi')
devices = sd.query_devices()
wasapi_devices = [(i, d['name'], d['max_input_channels'], d['max_output_channels'])
                  for i, d in enumerate(devices) if d['hostapi'] == wasapi_index]
input_devices = [(i, name) for i, name, ch_in, _ in wasapi_devices if ch_in > 0]
output_devices = [(i, name) for i, name, _, ch_out in wasapi_devices if ch_out > 0]

input_ax = plt.axes([0.6, 0.18, 0.3, 0.08])
output_ax = plt.axes([0.6, 0.095, 0.3, 0.08])
rate_ax = plt.axes([0.6, 0.04, 0.12, 0.05])
button_ax = plt.axes([0.76, 0.04, 0.14, 0.05])

input_selector = RadioButtons(input_ax, [f"{i}: {name}" for i, name in input_devices], active=0)
output_selector = RadioButtons(output_ax, [f"{i}: {name}" for i, name in output_devices], active=0)
rate_selector = RadioButtons(rate_ax, [str(r) for r in sample_rates], active=sample_rates.index(current_fs[0]))
rate_selector.on_clicked(update_sample_rate)

for selector in [input_selector, output_selector, rate_selector]:
    for label in selector.labels:
        label.set_fontsize(8)

start_button = Button(button_ax, 'Start Audio')
start_button.on_clicked(start_audio)

update(None)
plt.show()
