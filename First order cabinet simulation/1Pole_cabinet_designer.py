# Modified Cabinet Simulator with Grouped Parallel Filters and Improved UI
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

sample_rates = [44100, 48000]
current_fs = [48000]
BLOCK_SIZE = 2**14
DATA_TYPE = 'int32'
LATENCY = 'high'
stream = None
frequencies = np.linspace(10, 22000, 2048)
filter_types = ['HPF', 'LPF', 'BPF', 'BSF']

block_time = BLOCK_SIZE / current_fs[0]
block_ms = block_time * 1000  # convert to milliseconds
print(f"Block size:  {BLOCK_SIZE} samples")
print(f"Sample rate: {current_fs[0]} Hz")
print(f"Audio block: {block_ms:.2f} ms")

# Default filter parameters
filters      = ['HPF', 'BPF',  'BPF',   'BPF',  'LPF',  'LPF']
default_fc   = [80,     120,    600,    2500,   5000,   8000 ]
default_bw   = [40,     80,     500,    1200,   1000,   2000 ]
default_gain = [6,      5,      -4,     6,      -2,     6    ]
FREQ_MIN = 20
FREQ_MAX = 20000

fig = plt.figure(figsize=(12, 8))
plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

ax_plot_main = fig.add_axes([0.42, 0.55, 0.55, 0.4])
ax_plot_stages = fig.add_axes([0.42, 0.20, 0.55, 0.25])
response_line, = ax_plot_main.semilogx(frequencies, np.zeros_like(frequencies), label="Combined")
stage_lines = [ax_plot_stages.semilogx(frequencies, np.zeros_like(frequencies), label=f"S{i}")[0] for i in range(6)]

# Combined plot
ax_plot_main.set_ylim(-40, 10)
ax_plot_main.set_xlim(20, 20000)
ax_plot_main.grid(True)
ax_plot_main.set_title("Speaker Simulation Frequency Response")
ax_plot_main.set_xlabel("Frequency (Hz)")
ax_plot_main.set_ylabel("Gain (dB)")

# Per-stage plot
ax_plot_stages.set_ylim(-40, 10)
ax_plot_stages.set_xlim(20, 20000)
ax_plot_stages.grid(True)
ax_plot_stages.set_title("Individual Filter Responses")
ax_plot_stages.set_xlabel("Frequency (Hz)")
ax_plot_stages.set_ylabel("Gain (dB)")
ax_plot_stages.legend(fontsize=8, loc="lower right")

class FilterStage:
    def __init__(self, index, y_base):
        self.enabled = True
        self.parallel = index in [1, 2, 3]
        self.type = filters[index]
        self.fc = default_fc[index]
        self.bw = default_bw[index]
        self.gain_db = default_gain[index]  # Default gain in dB

        # Filter type
        ax_type = fig.add_axes([0.02, y_base + 0.07, 0.05, 0.1])
        self.type_selector = RadioButtons(ax_type, filter_types, active=filter_types.index(self.type))
        self.type_selector.on_clicked(self.update_and_plot)

        # Enable
        ax_enable = fig.add_axes([0.07, y_base + 0.07, 0.04, 0.05])
        self.enable_box = CheckButtons(ax_enable, ['On'], [True])
        self.enable_box.on_clicked(self.toggle_enable)

        # Parallel
        ax_mode = fig.add_axes([0.07, y_base + 0.12, 0.04, 0.05])
        self.mode_box = CheckButtons(ax_mode, ['||'], [self.parallel])
        self.mode_box.on_clicked(self.toggle_parallel)

        # Frequency
        self.ax_fc = fig.add_axes([0.15, y_base + 0.125, 0.15, 0.025])
        self.fc_slider = Slider(self.ax_fc, 'Fc', np.log10(FREQ_MIN), np.log10(FREQ_MAX),
                                valinit=np.log10(self.fc), valfmt="")
        self.fc_slider.on_changed(self.update_and_plot)

        # Range
        self.ax_bw = fig.add_axes([0.15, y_base + 0.1, 0.15, 0.025])
        self.bw_slider = Slider(self.ax_bw, 'BW', np.log10(10), np.log10(FREQ_MAX // 2),
                                valinit=np.log10(self.bw), valfmt="")
        self.bw_slider.on_changed(self.update_and_plot)

        # Gain
        self.ax_gain = fig.add_axes([0.15, y_base + 0.075, 0.15, 0.025])
        self.gain_slider = Slider(self.ax_gain, 'Gain', -24, 12, valinit=self.gain_db, valstep=0.5)
        self.gain_slider.on_changed(self.update_and_plot)

        # Text label
        ax_txt = fig.add_axes([0.15, y_base + 0.043, 0.05, 0.05])
        ax_txt.axis("off")
        self.label = ax_txt.text(0, 0.5, "", fontsize=8)

        self.update_slider_visibility()
        

    def toggle_enable(self, val):
        self.enabled = self.enable_box.get_status()[0]
        update_plot(None)

    def toggle_parallel(self, val):
        self.parallel = self.mode_box.get_status()[0]
        update_plot(None)

    def update_slider_visibility(self):
        self.ax_fc.set_visible(True)
        self.ax_bw.set_visible(self.type_selector.value_selected in ['BPF', 'BSF'])

    def update_and_plot(self, val):
        self.type = self.type_selector.value_selected
        self.update_slider_visibility()

        # Update slider text
        fc = int(10 ** self.fc_slider.val)
        self.fc_slider.valtext.set_text(f"{fc} Hz")
        if self.type in ['BPF', 'BSF']:
            bw = int(10 ** self.bw_slider.val)
            self.bw_slider.valtext.set_text(f"{bw} Hz")
            self.gain_db = self.gain_slider.val

        update_plot(None)

    def get_coeffs(self):
        if not self.enabled:
            return None

        fc = 10 ** self.fc_slider.val
        bw = 10 ** self.bw_slider.val
        gain = 10 ** (self.gain_db / 20)

        if self.type == 'LPF' or self.type == 'HPF':
            a_q24 = fc_to_q24(fc, current_fs[0])
            self.label.set_text(f"A = 0x{a_q24:08X}")
            return one_pole_iir_q24(a_q24, 'low' if self.type == 'LPF' else 'high')

        elif self.type == 'BPF':
            fc_low = max(fc - bw / 2, FREQ_MIN)
            fc_high = min(fc + bw / 2, FREQ_MAX)
            a_low = fc_to_q24(fc_low, current_fs[0])
            a_high = fc_to_q24(fc_high, current_fs[0])
            self.label.set_text(f"L=0x{a_low:08X} H=0x{a_high:08X}")
            b1, a1 = one_pole_iir_q24(a_low, 'high')
            b2, a2 = one_pole_iir_q24(a_high, 'low')
            return ([b1, b2], [a1, a2])

        elif self.type == 'BSF':
            fc_low = max(fc - bw / 2, FREQ_MIN)
            fc_high = min(fc + bw / 2, FREQ_MAX)
            a_low = fc_to_q24(fc_low, current_fs[0])
            a_high = fc_to_q24(fc_high, current_fs[0])
            self.label.set_text(f"L=0x{a_low:08X} H=0x{a_high:08X}")
            b_lp, a_lp = one_pole_iir_q24(a_low, 'low')
            b_hp, a_hp = one_pole_iir_q24(a_high, 'high')
            return ([b_lp, b_hp], [a_lp, a_hp])
        
        gain = 10 ** (self.gain_db / 20)
        if self.type in ['LPF', 'HPF']:
            b, a = one_pole_iir_q24(a_q24, ...)
            b = [bi * gain for bi in b]
            return b, a

def apply_filter(stage, x, b, a):
    gain = 10 ** (stage.gain_slider.val / 20)

    if stage.type == 'BSF':
        y = lfilter(b[0], a[0], x) + lfilter(b[1], a[1], x)
    elif stage.type == 'BPF':
        y = lfilter(b[1], a[1], lfilter(b[0], a[0], x))
    else:
        y = lfilter(b, a, x)

    return gain * y  # Apply gain here

def update_plot(val):
    w = frequencies
    h_total = np.ones_like(w, dtype=np.complex64)
    group = []
    group_indices = []

    # Reset per-stage plots
    for line in stage_lines:
        line.set_ydata(np.zeros_like(frequencies))

    def combine_and_apply_group(h, group, group_indices):
        if not group:
            return h
        if len(group) == 1:
            h *= group[0]
        else:
            h_group = sum(group) / len(group)
            h *= h_group
        return h

    for i, stage in enumerate(stages):
        coeffs = stage.get_coeffs()
        if not coeffs:
            continue

        b, a = coeffs
        if stage.type in ['BSF', 'BPF']:
            _, h1 = freqz(b[0], a[0], worN=w, fs=current_fs[0])
            _, h2 = freqz(b[1], a[1], worN=w, fs=current_fs[0])
            h_stage = h1 + h2 if stage.type == 'BSF' else h1 * h2
        else:
            _, h_stage = freqz(b, a, worN=w, fs=current_fs[0])

        # ✅ Apply gain
        gain = 10 ** (stage.gain_slider.val / 20)
        h_stage *= gain

        # Draw individual response
        stage_lines[i].set_ydata(20 * np.log10(np.maximum(np.abs(h_stage), 1e-9)))
        stage_lines[i].set_alpha(1.0 if stage.enabled else 0.1)

        if stage.parallel:
            group.append(h_stage)
        else:
            if group:
                # End of parallel block: combine group
                if len(group) > 1:
                    h_total *= sum(group) / len(group)
                else:
                    h_total *= group[0]
                group = []
            h_total *= h_stage

        # If last stage is part of group, flush it
        if i == len(stages) - 1 and group:
            if len(group) > 1:
                h_total *= sum(group) / len(group)
            else:
                h_total *= group[0]

    response_line.set_ydata(20 * np.log10(np.maximum(np.abs(h_total), 1e-9)))
    fig.canvas.draw_idle()


# Create filter stage instances
stages = [FilterStage(i, 0.75 - i * 0.12) for i in range(6)]

# Update labels right after creation
for s in stages:
    s.update_and_plot(None)

update_plot(None)

# --- Audio I/O selection and stream setup ---
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
rate_selector.on_clicked(lambda label: update_sample_rate(int(label)))

for selector in [input_selector, output_selector, rate_selector]:
    for label in selector.labels:
        label.set_fontsize(8)

start_button = Button(button_ax, 'Start Audio')
start_button.on_clicked(lambda event: start_audio())

def update_sample_rate(rate):
    global stream
    current_fs[0] = rate
    if stream:
        stream.stop()
        stream.close()
    update_plot(None)

def start_audio():
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

def audio_callback(indata, outdata, frames, time, status):
    if status:
        print(status)

    for ch in range(2):
        x = indata[:, ch]
        group = []

        for i, stage in enumerate(stages):
            coeffs = stage.get_coeffs()
            if not coeffs:
                continue

            b, a = coeffs
            y = apply_filter(stage, x, b, a)

            if stage.parallel:
                group.append(y)
                # If next stage is not parallel or this is the last stage: flush
                is_last = (i == len(stages) - 1)
                next_is_parallel = (i + 1 < len(stages) and stages[i + 1].parallel)
                if not next_is_parallel or is_last:
                    x = sum(group) / len(group) if len(group) > 1 else group[0]
                    group = []
            else:
                # Flush group before applying series stage
                if group:
                    x = sum(group) / len(group) if len(group) > 1 else group[0]
                    group = []
                x = y  # Series stage directly updates signal

        outdata[:, ch] = np.clip(x, -2**31, 2**31 - 1)


plt.show()
