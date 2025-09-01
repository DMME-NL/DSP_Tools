import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons, Button
from scipy.signal import freqz, lfilter
import sounddevice as sd
import threading, math

# =================================
# --- Global audio configuration ---
# =================================
sample_rates = [44100, 48000]
current_fs = [48000]
stream = None

# Realtime-friendly defaults (tweak if needed)
BLOCK_SIZE  = 2**8          # larger block => fewer callbacks
DATA_TYPE   = 'float32'     # avoid int<->float thrash in Python
LATENCY     = 'high'   # try 'exclusive'; "high" if unsupported

print(f"Block size: {BLOCK_SIZE}  | Sample rate: {current_fs[0]} Hz  | Block {1000*BLOCK_SIZE/current_fs[0]:.2f} ms")

# =============================
# --- Helpers (C-style math) ---
# =============================
def alpha_from_hz(fc, fs):
    if fc <= 0.0: return 0.0
    a = 1.0 - math.exp(-2.0*math.pi*fc/fs)
    return max(0.0, min(1.0, a))

def db_to_gain(db):
    return 10.0**(db/20.0)

def clamp1(x):
    return np.clip(x, -1.0, +1.0)

class OnePole:
    """y += a*(x-y)  → LPF; HPF = x - LPF(x).
       Keeps L/R states. Supports per-sample and block modes."""
    def __init__(self, a=0.0):
        self.a = float(a)
        self.sL = 0.0
        self.sR = 0.0

    def set_alpha(self, a):
        self.a = float(a)

    # ---- per-sample API (kept for clarity) ----
    def lpf_L(self, x): self.sL += self.a*(x - self.sL); return self.sL
    def lpf_R(self, x): self.sR += self.a*(x - self.sR); return self.sR
    def hpf_L(self, x): l = self.lpf_L(x); return x - l
    def hpf_R(self, x): l = self.lpf_R(x); return x - l

    # ---- block API (fast) ----
    def _ba(self):
        # For y += a*(x-y): LPF TF is H(z) = a / (1 - (1-a) z^-1)
        return np.array([self.a], np.float32), np.array([1.0, -(1.0 - self.a)], np.float32)

    def lpf_block_L(self, x):
        if self.a <= 0.0 or x.size == 0:
            return np.full_like(x, self.sL, dtype=np.float32)
        b, a = self._ba()
        y, zf = lfilter(b, a, x.astype(np.float32, copy=False), zi=[self.sL])
        self.sL = float(zf[0])
        return y.astype(np.float32, copy=False)

    def lpf_block_R(self, x):
        if self.a <= 0.0 or x.size == 0:
            return np.full_like(x, self.sR, dtype=np.float32)
        b, a = self._ba()
        y, zf = lfilter(b, a, x.astype(np.float32, copy=False), zi=[self.sR])
        self.sR = float(zf[0])
        return y.astype(np.float32, copy=False)

    def hpf_block_L(self, x):
        l = self.lpf_block_L(x)
        return (x - l).astype(np.float32, copy=False)

    def hpf_block_R(self, x):
        l = self.lpf_block_R(x)
        return (x - l).astype(np.float32, copy=False)

# =========================
# --- Nonlinear sections ---
# =========================
def triode_ws_35_asym(x, k3_pos, k5_pos, k3_neg, k5_neg, x5_gate=0.08):
    """
    Odd polynomial waveshaper with per-sign coefficients:
        y = x - k3*x^3 - k5*x^5
    5th order term is gated in for |x| >= x5_gate.
    Accepts scalar or array 'k3_neg/k5_neg' for per-sample asymmetry.
    """
    x = np.asarray(x, dtype=np.float32)
    x = clamp1(x)
    pos = (x >= 0.0)

    # broadcast scalars/arrays to match x
    k3p = np.broadcast_to(np.asarray(k3_pos, dtype=np.float32), x.shape)
    k5p = np.broadcast_to(np.asarray(k5_pos, dtype=np.float32), x.shape)
    k3n = np.broadcast_to(np.asarray(k3_neg, dtype=np.float32), x.shape)
    k5n = np.broadcast_to(np.asarray(k5_neg, dtype=np.float32), x.shape)

    k3 = np.where(pos, k3p, k3n)
    k5 = np.where(pos, k5p, k5n)

    use5 = (np.abs(x) >= float(x5_gate)).astype(np.float32)
    k5 = k5 * use5

    x3 = x*x*x
    x5 = x3*x*x
    y = x - k3*x3 - k5*x5
    return clamp1(y)

def cathode_squish(x, amount=0.22, neg_hold=0.97):
    x = np.asarray(x, dtype=np.float32)
    y = x.copy()
    pos = y > 0.0
    y[pos] = y[pos] - amount*(y[pos]*y[pos])
    y[~pos] *= neg_hold
    return clamp1(y)

# ==========================
# --- Header configuration ---
# ==========================
# Processing options (match the .h defaults)
JCM_ECO      = 1   # 1 = skip post-LPF
JCM_USE_X5   = 1   # use x^5 term (gated)

# Staging
JCM_INPUT_PAD_DB     = +10.0
JCM_STAGEA_GAIN_DB   = +10.0
JCM_STAGEB_GAIN_DB   = +12.0
JCM_STACK_MAKEUP_DB  = +14.0

# Pot mappings (dB ranges)
JCM_PREVOL_MIN_DB    = -40.0   # +2 dB top boost near max
JCM_PREVOL_TAPER     = 1.35
JCM_PREVOL_TOP_BOOST_DB = +2.0
JCM_PRESENCE_MAX_DB  = +8.0
JCM_MASTER_MIN_DB    = -40.0
JCM_MASTER_MAX_DB    = +0.0

# Waveshaper coefficients
JCM_K3A, JCM_K5A = 0.28, 0.08
JCM_K3B, JCM_K5B = 0.45, 0.15
ASYM_A_BASE, ASYM_A_DEPTH = 0.70, 0.00     # Stage A asym (fixed)
ASYM_B_BASE, ASYM_B_DEPTH = 0.62, 0.08     # Stage B asym (env-tracked)

WS_X5_ON     = 0.08    # 5th term gate threshold
JCM_ENVB_HZ  = 12.0    # env LPF (~12 Hz)
JCM_ENV_DECIM= 2       # update every other sample

# Voice / filter corners (from .h)
class Voice:
    pre_hpf_Hz   = 20.0    # gentle rumble control
    cpl1_hz      = 12.0
    cpl2_hz      = 40.0
    bright_hz_min= 2500.0
    bright_hz_max= 8000.0
    bass_hz      = 100.0
    mid_hz       = 650.0
    treble_hz    = 4500.0
    presence_hz  = 3500.0
    post_lpf_Hz  = 12000.0

# ============================
# --- Main Preamp class    ---
# ============================
class JCM800Preamp:
    def __init__(self, fs=48000):
        self.fs = fs
        self.voice = Voice()

        # Pots (0..1), matching the .h file names/behavior
        self.p_prevol = 0.5
        self.p_bass   = 0.5
        self.p_mid    = 0.5
        self.p_treble = 0.5
        self.p_pres   = 0.5
        self.p_master = 0.5

        # Fixed gains / stages (from .h)
        self.input_pad   = db_to_gain(JCM_INPUT_PAD_DB)
        self.stageA_gain = db_to_gain(JCM_STAGEA_GAIN_DB)
        self.stageB_gain = db_to_gain(JCM_STAGEB_GAIN_DB)
        self.stack_makeup= db_to_gain(JCM_STACK_MAKEUP_DB)

        # Nonlinearity coefficients
        self.k3A, self.k5A = JCM_K3A, JCM_K5A
        self.k3B, self.k5B = JCM_K3B, JCM_K5B
        self.asymA         = ASYM_A_BASE

        # Envelope for Stage B asymmetry
        self.envB    = 0.0
        self.env_a   = alpha_from_hz(JCM_ENVB_HZ, fs)
        self.env_phase = 0  # 0 or 1, for decimation alignment across blocks

        # Filters
        self.preHPF   = OnePole(alpha_from_hz(self.voice.pre_hpf_Hz, fs))
        self.cpl1     = OnePole(alpha_from_hz(self.voice.cpl1_hz,     fs))
        self.brightLP = OnePole(alpha_from_hz(self.voice.bright_hz_max, fs))  # fc updated by prevol
        self.cpl2     = OnePole(alpha_from_hz(self.voice.cpl2_hz,     fs))
        self.bassLP   = OnePole(alpha_from_hz(self.voice.bass_hz,     fs))
        self.midHP    = OnePole(alpha_from_hz(self.voice.mid_hz,      fs))
        self.midLP    = OnePole(alpha_from_hz(self.voice.mid_hz,      fs))
        self.trebLP   = OnePole(alpha_from_hz(self.voice.treble_hz,   fs))
        self.presLP   = OnePole(alpha_from_hz(self.voice.presence_hz, fs))
        self.postLP   = OnePole(alpha_from_hz(self.voice.post_lpf_Hz, fs))

        # Mapped parameters (derived from pots)
        self.prevol       = 1.0
        self.bright_mix   = 0.0
        self.bass_gain    = 1.0
        self.mid_gain     = 1.0
        self.treble_gain  = 1.0
        self.presence_gain= 1.0
        self.master       = 1.0

        self._update_params()

    # ----- pot mappings (identical to .h behavior) -----
    def _update_params(self):
        # PreVol (−40 dB → 0 dB with audio taper + 2 dB top boost)
        p = float(self.p_prevol)
        t = p**JCM_PREVOL_TAPER
        prevol_db = JCM_PREVOL_MIN_DB + (0.0 - JCM_PREVOL_MIN_DB)*t
        prevol_db += JCM_PREVOL_TOP_BOOST_DB * (p**6.0)
        self.prevol = db_to_gain(prevol_db)

        # Bright mix fades out with PreVol; corner moves 8k → 2.5k as PreVol increases
        inv01 = 1.0 - t
        self.bright_mix = inv01 * (db_to_gain(4.0) - 1.0)  # +4 dB cap in .h
        bright_fc = self.voice.bright_hz_min + (self.voice.bright_hz_max - self.voice.bright_hz_min)*(1.0 - p)
        self.brightLP.set_alpha(alpha_from_hz(bright_fc, self.fs))

        # Tone pots (dB ranges from .h)
        self.bass_gain    = db_to_gain(-12.0 + self.p_bass  * ( +6.0 + 12.0))
        self.mid_gain     = db_to_gain(-12.0 + self.p_mid   * ( +12.0 + 12.0))
        self.treble_gain  = db_to_gain(-12.0 + self.p_treble* ( +6.0 + 12.0))
        self.presence_gain= db_to_gain(  0.0 + self.p_pres  * ( JCM_PRESENCE_MAX_DB))
        self.master       = db_to_gain( JCM_MASTER_MIN_DB + self.p_master*(JCM_MASTER_MAX_DB - JCM_MASTER_MIN_DB) )

    def set_sr(self, fs):
        self.fs = fs
        self.preHPF.set_alpha(alpha_from_hz(self.voice.pre_hpf_Hz, fs))
        self.cpl1  .set_alpha(alpha_from_hz(self.voice.cpl1_hz, fs))
        self.cpl2  .set_alpha(alpha_from_hz(self.voice.cpl2_hz, fs))
        self.bassLP.set_alpha(alpha_from_hz(self.voice.bass_hz, fs))
        self.midHP .set_alpha(alpha_from_hz(self.voice.mid_hz,  fs))
        self.midLP .set_alpha(alpha_from_hz(self.voice.mid_hz,  fs))
        self.trebLP.set_alpha(alpha_from_hz(self.voice.treble_hz, fs))
        self.presLP.set_alpha(alpha_from_hz(self.voice.presence_hz, fs))
        self.postLP.set_alpha(alpha_from_hz(self.voice.post_lpf_Hz, fs))
        self.env_a = alpha_from_hz(JCM_ENVB_HZ, fs)
        self._update_params()

    # ---- I/O scaling ----
    def _convert_in(self, x):
        if x.ndim == 1: x = x[:, None]
        if DATA_TYPE == 'float32':
            return x.astype(np.float32, copy=False)
        elif DATA_TYPE == 'int32':
            return (x.astype(np.float32, copy=False) / 2147483648.0)
        elif DATA_TYPE == 'int16':
            return (x.astype(np.float32, copy=False) / 32768.0)
        elif DATA_TYPE == 'uint8':
            return ((x.astype(np.float32, copy=False) - 128.0) / 128.0)
        else:
            return x.astype(np.float32, copy=False)

    def _convert_out(self, y):
        y = clamp1(y)
        if DATA_TYPE == 'float32':
            return y.astype(np.float32, copy=False)
        elif DATA_TYPE == 'int32':
            return (y*2147483647.0).astype(np.int32, copy=False)
        elif DATA_TYPE == 'int16':
            return (y*32767.0).astype(np.int16, copy=False)
        elif DATA_TYPE == 'uint8':
            return (y*127.0 + 128.0).astype(np.uint8, copy=False)
        else:
            return y.astype(np.float32, copy=False)

    # ---- Core processing (vectorized; matches .h ordering & pots) ----
    def process_block(self, x, bypass_nl=False):
        xf = self._convert_in(x)
        L = xf[:, 0].astype(np.float32, copy=False)
        R = (xf[:, 1] if xf.shape[1] > 1 else L).astype(np.float32, copy=False)

        # Input pad
        L *= self.input_pad
        R *= self.input_pad

        # Pre HPF (gentle)
        L = self.preHPF.hpf_block_L(L)
        R = self.preHPF.hpf_block_R(R)

        # Coupler #1 (into pre-vol)
        L = self.cpl1.hpf_block_L(L)
        R = self.cpl1.hpf_block_R(R)

        # PreVol gain & bright shelf (x + bright_mix * high)
        if abs(self.bright_mix) > 1e-6:
            lL = self.brightLP.lpf_block_L(L);  hL = L - lL
            lR = self.brightLP.lpf_block_R(R);  hR = R - lR
            L = (L + self.bright_mix*hL) * self.prevol
            R = (R + self.bright_mix*hR) * self.prevol
        else:
            L = L * self.prevol
            R = R * self.prevol

        # Stage A gain + nonlinearity
        if bypass_nl:
            L = clamp1(L * self.stageA_gain)
            R = clamp1(R * self.stageA_gain)
        else:
            k3A_neg = self.k3A * self.asymA
            k5A_neg = self.k5A * self.asymA
            L = triode_ws_35_asym(L * self.stageA_gain,
                                  self.k3A, self.k5A if JCM_USE_X5 else 0.0,
                                  k3A_neg, k5A_neg if JCM_USE_X5 else 0.0, WS_X5_ON)
            R = triode_ws_35_asym(R * self.stageA_gain,
                                  self.k3A, self.k5A if JCM_USE_X5 else 0.0,
                                  k3A_neg, k5A_neg if JCM_USE_X5 else 0.0, WS_X5_ON)

        # Coupler #2 (into Stage B)
        L = self.cpl2.hpf_block_L(L)
        R = self.cpl2.hpf_block_R(R)

        # Envelope for Stage B asymmetry (decimated by 2, forward-hold)
        s_abs = 0.5*(np.abs(L) + np.abs(R))
        n = s_abs.shape[0]
        env = np.empty_like(s_abs, dtype=np.float32)
        phase = self.env_phase
        idxs = np.arange(phase, n, JCM_ENV_DECIM, dtype=int)
        if idxs.size:
            b = np.array([self.env_a], np.float32)
            a = np.array([1.0, -(1.0 - self.env_a)], np.float32)
            y_sub, _ = lfilter(b, a, s_abs[idxs].astype(np.float32, copy=False), zi=[self.envB])
            # forward fill: hold last updated value
            current = float(self.envB)
            k = 0
            next_idx = idxs[0]
            for i in range(n):
                if k < idxs.size and i == next_idx:
                    current = float(y_sub[k])
                    k += 1
                    next_idx = idxs[k] if k < idxs.size else n + 1
                env[i] = current
            self.envB = float(current)
        else:
            env[:] = float(self.envB)
        self.env_phase = (phase + n) & 1  # keep alignment across blocks

        # Stage B gain + nonlinearity (env-tracked asym on negative lobe)
        if bypass_nl:
            L = clamp1(L * self.stageB_gain)
            R = clamp1(R * self.stageB_gain)
        else:
            asymB = ASYM_B_BASE + ASYM_B_DEPTH * env
            k3B_neg = self.k3B * asymB
            k5B_neg = self.k5B * asymB
            L = triode_ws_35_asym(L * self.stageB_gain,
                                  self.k3B, self.k5B if JCM_USE_X5 else 0.0,
                                  k3B_neg, k5B_neg if JCM_USE_X5 else 0.0, WS_X5_ON)
            R = triode_ws_35_asym(R * self.stageB_gain,
                                  self.k3B, self.k5B if JCM_USE_X5 else 0.0,
                                  k3B_neg, k5B_neg if JCM_USE_X5 else 0.0, WS_X5_ON)

        # Cathode follower-ish squish
        if not bypass_nl:
            L = cathode_squish(L, 0.22, 0.97)
            R = cathode_squish(R, 0.22, 0.97)

        # Tone stack proxy: low/mid/high
        lowL  = self.bassLP.lpf_block_L(L) * self.bass_gain
        lowR  = self.bassLP.lpf_block_R(R) * self.bass_gain

        midHL = self.midHP.hpf_block_L(L)
        midHR = self.midHP.hpf_block_R(R)
        midL  = self.midLP.lpf_block_L(midHL) * self.mid_gain
        midR  = self.midLP.lpf_block_R(midHR) * self.mid_gain

        tLL   = self.trebLP.lpf_block_L(L)
        tLR   = self.trebLP.lpf_block_R(R)
        highL = (L - tLL) * self.treble_gain
        highR = (R - tLR) * self.treble_gain

        mixL = (lowL + midL + highL) * self.stack_makeup
        mixR = (lowR + midR + highR) * self.stack_makeup

        # Presence shelf (post-stack high shelf)
        if abs(self.presence_gain - 1.0) > 1e-6:
            pLL = self.presLP.lpf_block_L(mixL)
            pLR = self.presLP.lpf_block_R(mixR)
            mixL = mixL + (self.presence_gain - 1.0) * (mixL - pLL)
            mixR = mixR + (self.presence_gain - 1.0) * (mixR - pLR)

        # Post LPF (skipped in ECO=1)
        if not JCM_ECO:
            mixL = self.postLP.lpf_block_L(mixL)
            mixR = self.postLP.lpf_block_R(mixR)

        y = np.stack([mixL, mixR], axis=-1) * self.master
        return self._convert_out(y)

    # ---------- FR helpers ----------
    @staticmethod
    def _ba_from_alpha(a):
        return np.array([a], float), np.array([1.0, -(1.0 - a)], float)

    def fr_linear_derived(self, worN=2048):
        """Analytical FR (nonlinears removed), composed like the .h chain."""
        fs = self.fs
        _, preLP  = freqz(*self._ba_from_alpha(self.preHPF.a), worN=worN, fs=fs)
        _, cpl1LP = freqz(*self._ba_from_alpha(self.cpl1.a),   worN=worN, fs=fs)
        _, brLP   = freqz(*self._ba_from_alpha(self.brightLP.a), worN=worN, fs=fs)
        _, cpl2LP = freqz(*self._ba_from_alpha(self.cpl2.a),   worN=worN, fs=fs)
        _, bassLP = freqz(*self._ba_from_alpha(self.bassLP.a), worN=worN, fs=fs)
        _, midLP  = freqz(*self._ba_from_alpha(self.midLP.a),  worN=worN, fs=fs)
        _, midHPLP= freqz(*self._ba_from_alpha(self.midHP.a),  worN=worN, fs=fs)
        _, trebLP = freqz(*self._ba_from_alpha(self.trebLP.a), worN=worN, fs=fs)
        _, presLP = freqz(*self._ba_from_alpha(self.presLP.a), worN=worN, fs=fs)
        _, postLP = freqz(*self._ba_from_alpha(self.postLP.a), worN=worN, fs=fs)

        H_preHPF = (1.0 - preLP)
        H_cpl1HPF= (1.0 - cpl1LP)
        H_bright = 1.0 + self.bright_mix*(1.0 - brLP)
        H_cpl2HPF= (1.0 - cpl2LP)

        H_low  = bassLP * self.bass_gain
        H_mid  = midLP * (1.0 - midHPLP) * self.mid_gain
        H_high = (1.0 - trebLP) * self.treble_gain
        H_tone = H_low + H_mid + H_high

        H_pres = 1.0 + (self.presence_gain - 1.0)*(1.0 - presLP)
        H_post = postLP if not JCM_ECO else 1.0

        G = self.input_pad * self.prevol * self.stageA_gain * self.stageB_gain * self.stack_makeup * self.master
        H_total = (H_preHPF * H_cpl1HPF * H_bright * H_cpl2HPF) * H_tone * H_pres * H_post * G
        f = np.linspace(0, fs/2, worN)
        return f, H_total, (H_low, H_mid, H_high, H_preHPF, H_post)

    def fr_measured(self, worN=2048, bypass_nl=False):
        fs = self.fs
        N = 1<<15
        x = (np.random.randn(N,2)*0.2).astype(np.float32)  # small amplitude
        # ensure float path for FFT
        global DATA_TYPE
        old_dtype = DATA_TYPE
        DATA_TYPE = 'float32'
        try:
            y = self.process_block(x, bypass_nl=bypass_nl).astype(np.float32)
        finally:
            DATA_TYPE = old_dtype

        X = np.fft.rfft(x[:,0])
        Y = np.fft.rfft(y[:,0])
        eps = 1e-12
        H = Y / np.maximum(np.abs(X), eps)
        f = np.fft.rfftfreq(N, 1.0/fs)

        if len(H) != worN:
            idx = np.linspace(0, len(H)-1, worN)
            Hr = np.interp(idx, np.arange(len(H)), H.real)
            Hi = np.interp(idx, np.arange(len(H)), H.imag)
            H = Hr + 1j*Hi
            f = np.linspace(0, fs/2, worN)
        return f, H

# ======================================
# --- UI / Audio / Plot glue code ---
# ======================================
preamp = JCM800Preamp(fs=current_fs[0])
stream_lock = threading.Lock()

def dB(x): return 20*np.log10(np.maximum(np.abs(x), 1e-12))

# --- plotting canvas (Total FR + components) ---
fig, (ax1, ax2) = plt.subplots(2,1, figsize=(10,7), sharex=True)
plt.subplots_adjust(left=0.08, right=0.78, top=0.95, bottom=0.28, hspace=0.2)

f_init = np.linspace(20, current_fs[0]//2, 2048)
line_total = ax1.semilogx(f_init, np.full_like(f_init, -120.0), label="Total FR", color='C0')[0]
ax1.set_title("Marshall JCM800 – Total Frequency Response")
ax1.set_ylabel("dB"); ax1.grid(True, which='both'); ax1.set_ylim(-30, 5)

comp_low  = ax2.semilogx(f_init, np.full_like(f_init, -120.0), '--', label="Low path")[0]
comp_mid  = ax2.semilogx(f_init, np.full_like(f_init, -120.0), '--', label="Mid path")[0]
comp_high = ax2.semilogx(f_init, np.full_like(f_init, -120.0), '--', label="High path")[0]
comp_pre  = ax2.semilogx(f_init, np.full_like(f_init, -120.0), ':',  label="Pre HPF")[0]
comp_post = ax2.semilogx(f_init, np.full_like(f_init, -120.0), ':',  label="Post LPF")[0]
ax2.set_title("Sections (analytical)"); ax2.set_xlabel("Hz"); ax2.set_ylabel("dB")
ax2.grid(True, which='both'); ax2.set_xlim(20, 20000); ax2.set_ylim(-60, 12)
ax1.legend(); ax2.legend()

# --- mode toggle (analytical vs measured) ---
mode_ax = plt.axes([0.80, 0.42, 0.16, 0.12])
mode_sel = RadioButtons(mode_ax, ['Linear (analytical)', 'Measured (NL off)', 'Measured (NL on)'], active=0)
for l in mode_sel.labels: l.set_fontsize(8)

# redraw throttling (avoid UI vs audio contention)
_need_redraw = [False]
def _schedule_redraw():
    if not _need_redraw[0]:
        _need_redraw[0] = True
        t = fig.canvas.new_timer(interval=100)  # ~10 Hz
        t.add_callback(_do_redraw_once)
        t.start()
def _do_redraw_once():
    if _need_redraw[0]:
        _need_redraw[0] = False
        redraw()

def redraw():
    fs = current_fs[0]
    mode = mode_sel.value_selected
    worN = 2048

    if mode == 'Linear (analytical)':
        f, H, (H_low, H_mid, H_high, H_preHPF, H_post) = preamp.fr_linear_derived(worN=worN)

        # --- normalize everything relative to total FR peak ---
        norm = np.max(np.abs(H))
        if norm <= 0: norm = 1.0
        H       = H / norm
        H_low   = H_low / norm
        H_mid   = H_mid / norm
        H_high  = H_high / norm
        H_preHPF= H_preHPF / norm
        H_post  = H_post / norm

        line_total.set_ydata(dB(H))
        comp_low .set_ydata(dB(H_low))
        comp_mid .set_ydata(dB(H_mid))
        comp_high.set_ydata(dB(H_high))
        comp_pre .set_ydata(dB(H_preHPF))
        comp_post.set_ydata(dB(H_post if not np.isscalar(H_post) else np.ones_like(H)*H_post))

        for ln in (line_total, comp_low, comp_mid, comp_high, comp_pre, comp_post):
            ln.set_xdata(f)
    else:
        bypass_nl = (mode == 'Measured (NL off)')
        f, H = preamp.fr_measured(worN=worN, bypass_nl=bypass_nl)
        line_total.set_ydata(dB(H))
        line_total.set_xdata(f)
        # hide components in measured modes
        for ln in (comp_low, comp_mid, comp_high, comp_pre, comp_post):
            ln.set_ydata(np.full_like(f, np.nan))
            ln.set_xdata(f)

    fig.canvas.draw_idle()

mode_sel.on_clicked(lambda _: _schedule_redraw())

# --- pots (0..1) matching .h ---
slider_y_start = 0.21; slider_h = 0.02; slider_dy = 0.025
def make_slider(name, init, idx):
    y = slider_y_start - idx*slider_dy
    ax = plt.axes([0.10, y, 0.32, slider_h])
    return Slider(ax, name, 0.0, 1.0, valinit=init)

s_pre   = make_slider("PreVol",   preamp.p_prevol,  0)
s_bass  = make_slider("Bass",     preamp.p_bass,    1)
s_mid   = make_slider("Mid",      preamp.p_mid,     2)
s_treb  = make_slider("Treble",   preamp.p_treble,  3)
s_pres  = make_slider("Presence", preamp.p_pres,    4)
s_mast  = make_slider("Master",   preamp.p_master,  5)

def on_pot(_):
    preamp.p_prevol = s_pre.val
    preamp.p_bass   = s_bass.val
    preamp.p_mid    = s_mid.val
    preamp.p_treble = s_treb.val
    preamp.p_pres   = s_pres.val
    preamp.p_master = s_mast.val
    preamp._update_params()
    _schedule_redraw()

for s in (s_pre, s_bass, s_mid, s_treb, s_pres, s_mast):
    s.on_changed(on_pot)

# --- Audio I/O & device selectors ---
def update_sample_rate(label):
    global stream
    current_fs[0] = int(label)
    with stream_lock:
        preamp.set_sr(current_fs[0])
        if stream:
            try:
                stream.stop(); stream.close()
            finally:
                pass
            stream = None
    _schedule_redraw()

def audio_callback(indata, outdata, frames, time, status):
    if status:
        # under/overruns show up here
        pass
    # keep callback tiny; no locks
    outdata[:] = preamp.process_block(indata, bypass_nl=False)

def start_audio(_event):
    global stream
    with stream_lock:
        if stream:
            try: stream.stop(); stream.close()
            except: pass
            stream = None
        # Build device lists fresh in case hardware changed
        in_idx  = int(input_selector.value_selected.split(':')[0])
        out_idx = int(output_selector.value_selected.split(':')[0])
        stream = sd.Stream(
            samplerate=current_fs[0],
            blocksize=BLOCK_SIZE,
            dtype=DATA_TYPE,
            channels=2,
            latency=LATENCY,
            callback=audio_callback,
            device=(in_idx, out_idx)
        )
        stream.start()
    print(f"Audio started @ {current_fs[0]} Hz | block {BLOCK_SIZE} ({1000*BLOCK_SIZE/current_fs[0]:.2f} ms)")
    print(f"Input : {sd.query_devices(in_idx)['name']}")
    print(f"Output: {sd.query_devices(out_idx)['name']}")

# Build device lists (prefer WASAPI on Windows if present)
try:
    wasapi_index = next(i for i, api in enumerate(sd.query_hostapis()) if api['name'].lower() == 'windows wasapi')
except StopIteration:
    wasapi_index = None

devices = sd.query_devices()
if wasapi_index is not None:
    devs = [(i, d['name'], d['max_input_channels'], d['max_output_channels'])
            for i, d in enumerate(devices) if d['hostapi'] == wasapi_index]
else:
    devs = [(i, d['name'], d['max_input_channels'], d['max_output_channels'])
            for i, d in enumerate(devices)]

input_devices  = [(i, name) for i, name, ch_in, _ in devs if ch_in > 0]
output_devices = [(i, name) for i, name, _, ch_out in devs if ch_out > 0]

input_ax  = plt.axes([0.45, 0.15, 0.30, 0.09])
output_ax = plt.axes([0.45, 0.04, 0.30, 0.09])
rate_ax   = plt.axes([0.80, 0.30, 0.16, 0.08])
button_ax = plt.axes([0.80, 0.22, 0.16, 0.06])

input_selector  = RadioButtons(input_ax,  [f"{i}: {name}" for i, name in input_devices]  or ["0: Default In"],  active=0)
output_selector = RadioButtons(output_ax, [f"{i}: {name}" for i, name in output_devices] or ["0: Default Out"], active=0)
rate_selector   = RadioButtons(rate_ax,   [str(r) for r in sample_rates],
                               active=sample_rates.index(current_fs[0]) if current_fs[0] in sample_rates else 0)
rate_selector.on_clicked(update_sample_rate)
for sel in (input_selector, output_selector, rate_selector):
    for l in sel.labels: l.set_fontsize(8)

start_button = Button(button_ax, 'Start Audio')
start_button.on_clicked(start_audio)

# Initial draw
redraw()
plt.show()
