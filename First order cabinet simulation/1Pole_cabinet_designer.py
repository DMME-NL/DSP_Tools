
# Combined Speaker Simulation Tool with Unified UI and Logarithmic Sliders
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons, Button, CheckButtons
from scipy.signal import freqz, lfilter
import sounddevice as sd

# Fixed-point Q8.24 conversion
def q24_to_float(q): return q / (1 << 24)
def float_to_q24(f): return int(round(f * (1 << 24)))
def fc_to_q24(fc, fs): return float_to_q24(1 - np.exp(-2 * np.pi * fc / fs))
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

# Globals
sample_rates = [44100, 48000]
current_fs = [48000]

# Audio Stream Settings
BLOCK_SIZE  = pow(2, 14)    # 2^6(64) | 2*8(256) 
DATA_TYPE   ='int32'        # float32 | int16    | int32 | uint8 | float64
LATENCY     ='high'         # low     | high     | exclusive

stream = None
frequencies = np.linspace(10, 22000, 2048)
filter_types = ['LPF', 'HPF', 'BPF', 'BSF']

FREQ_MIN = 20
FREQ_MAX = 20000

fig = plt.figure(figsize=(12, 8))
plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

# --- Plot ---
ax_plot = fig.add_axes([0.38, 0.25, 0.58, 0.7])
response_line, = ax_plot.semilogx(frequencies, np.zeros_like(frequencies))
ax_plot.set_ylim(-60, 6)
ax_plot.set_xlim(20, 20000)
ax_plot.grid(True)
ax_plot.set_title("Speaker Simulation Frequency Response")
ax_plot.set_xlabel("Frequency (Hz)")
ax_plot.set_ylabel("Gain (dB)")

# --- Filter Stage ---
class FilterStage:
    def __init__(self, index, y_base):
        self.enabled = False
        if index is 0:
            self.type = 'LPF'
        elif index is 1:
            self.type = 'HPF'
        elif index is 2:
            self.type = 'BPF'

        self.fc = 1000
        self.fc_low = 300
        self.fc_high = 3000

        ax_type = fig.add_axes([0.02, y_base+0.07, 0.05, 0.1])
        self.type_selector = RadioButtons(ax_type, filter_types, active=index)
        self.type_selector.on_clicked(self.update_and_plot)

        ax_enable = fig.add_axes([0.07, y_base + 0.07, 0.04, 0.05])
        self.enable_box = CheckButtons(ax_enable, ['On'], [False])
        self.enable_box.on_clicked(self.toggle_enable)

        self.ax_fc1 = fig.add_axes([0.13, y_base + 0.125, 0.15, 0.025])
        self.fc1_slider = Slider(self.ax_fc1, 'L', np.log10(FREQ_MIN), np.log10(FREQ_MAX), valinit=np.log10(self.fc))
        self.fc1_slider.on_changed(self.update_and_plot)

        self.ax_fc2 = fig.add_axes([0.13, y_base + 0.1, 0.15, 0.025])
        self.fc2_slider = Slider(self.ax_fc2, 'H', np.log10(100), np.log10(FREQ_MAX), valinit=np.log10(self.fc_high))
        self.fc2_slider.on_changed(self.update_and_plot)

        ax_txt = fig.add_axes([0.13, y_base + 0.045, 0.05, 0.05])
        ax_txt.axis("off")
        self.label = ax_txt.text(0, 0.5, "", fontsize=8)

        self.update_slider_visibility()

    def toggle_enable(self, val):
        self.enabled = self.enable_box.get_status()[0]
        update_plot(None)
        self.update_slider_visibility()

    def update_slider_visibility(self):
        if self.type_selector.value_selected in ['LPF', 'HPF']:
            self.ax_fc1.set_visible(True)
            self.ax_fc2.set_visible(False)
        else:
            self.ax_fc1.set_visible(True)
            self.ax_fc2.set_visible(True)

    def update_and_plot(self, val):
        self.type = self.type_selector.value_selected
        self.update_slider_visibility()
        update_plot(None)

    def get_coeffs(self):
        if not self.enabled:
            return None
        t = self.type
        if t in ['LPF', 'HPF']:
            fc = 10 ** self.fc1_slider.val
            a_q24 = fc_to_q24(fc, current_fs[0])
            self.label.set_text(f"A = 0x{a_q24:08X}")
            return one_pole_iir_q24(a_q24, 'low' if t == 'LPF' else 'high')
        elif t == 'BPF':
            fc_low = 10 ** self.fc1_slider.val
            fc_high = 10 ** self.fc2_slider.val
            a_low = fc_to_q24(fc_low, current_fs[0])
            a_high = fc_to_q24(fc_high, current_fs[0])
            self.label.set_text(f"L=0x{a_low:08X} H=0x{a_high:08X}")
            b1, a1 = one_pole_iir_q24(a_low, 'high')
            b2, a2 = one_pole_iir_q24(a_high, 'low')
            return ([b1, b2], [a1, a2])
        elif t == 'BSF':
            fc_low = 10 ** self.fc1_slider.val
            fc_high = 10 ** self.fc2_slider.val
            a_low = fc_to_q24(fc_low, current_fs[0])
            a_high = fc_to_q24(fc_high, current_fs[0])
            self.label.set_text(f"L=0x{a_low:08X} H=0x{a_high:08X}")
            b_lp, a_lp = one_pole_iir_q24(a_low, 'low')
            b_hp, a_hp = one_pole_iir_q24(a_high, 'high')
            return ([b_lp, b_hp], [a_lp, a_hp])

stages = [
    FilterStage(0, 0.65),
    FilterStage(1, 0.5),
    FilterStage(2, 0.35),
]

def update_plot(val):
    w = frequencies
    h = np.ones_like(w, dtype=np.complex64)
    for stage in stages:
        coeffs = stage.get_coeffs()
        if coeffs:
            b, a = coeffs
            if stage.type == 'BSF':
                _, h_lp = freqz(b[0], a[0], worN=w, fs=current_fs[0])
                _, h_hp = freqz(b[1], a[1], worN=w, fs=current_fs[0])
                h *= h_lp + h_hp
            elif stage.type == 'BPF':
                _, h1 = freqz(b[0], a[0], worN=w, fs=current_fs[0])
                _, h2 = freqz(b[1], a[1], worN=w, fs=current_fs[0])
                h *= h1 * h2
            else:
                _, hi = freqz(b, a, worN=w, fs=current_fs[0])
                h *= hi
    response_line.set_ydata(20 * np.log10(np.maximum(np.abs(h), 1e-9)))
    fig.canvas.draw_idle()

def audio_callback(indata, outdata, frames, time, status):
    if status: print(status)
    for ch in range(2):
        x = indata[:, ch]
        for stage in stages:
            coeffs = stage.get_coeffs()
            if coeffs:
                b, a = coeffs
                if stage.type == 'BSF':
                    x_lp = lfilter(b[0], a[0], x)
                    x_hp = lfilter(b[1], a[1], x)
                    x = x_lp + x_hp
                elif stage.type == 'BPF':
                    x = lfilter(b[0], a[0], x)
                    x = lfilter(b[1], a[1], x)
                else:
                    x = lfilter(b, a, x)
        outdata[:, ch] = x

def update_sample_rate(label):
    global stream
    current_fs[0] = int(label)
    if stream:
        stream.stop()
        stream.close()
    update_plot(None)

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

# Audio I/O selection
wasapi_index = next(i for i, api in enumerate(sd.query_hostapis()) if api['name'].lower() == 'windows wasapi')
devices = sd.query_devices()
wasapi_devices = [(i, d['name'], d['max_input_channels'], d['max_output_channels'])
                  for i, d in enumerate(devices) if d['hostapi'] == wasapi_index]
input_devices = [(i, name) for i, name, ch_in, _ in wasapi_devices if ch_in > 0]
output_devices = [(i, name) for i, name, _, ch_out in wasapi_devices if ch_out > 0]

input_ax = fig.add_axes([0.1, 0.1, 0.4, 0.06])
output_ax = fig.add_axes([0.1, 0.03, 0.4, 0.06])
rate_ax = fig.add_axes([0.55, 0.03, 0.1, 0.06])
button_ax = fig.add_axes([0.7, 0.03, 0.2, 0.06])

input_selector = RadioButtons(input_ax, [f"{i}: {name}" for i, name in input_devices], active=0)
output_selector = RadioButtons(output_ax, [f"{i}: {name}" for i, name in output_devices], active=0)
rate_selector = RadioButtons(rate_ax, [str(r) for r in sample_rates], active=sample_rates.index(current_fs[0]))
rate_selector.on_clicked(update_sample_rate)

for selector in [input_selector, output_selector, rate_selector]:
    for label in selector.labels:
        label.set_fontsize(8)

start_button = Button(button_ax, 'Start Audio')
start_button.on_clicked(start_audio)

update_plot(None)
plt.show()
