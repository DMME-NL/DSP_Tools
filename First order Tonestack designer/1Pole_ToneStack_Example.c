#define PEAK_MAX        0x7FFFFF00      // Largest 24-bit sample for peak detection (~24-bit max)
#define PEAK_MIN       -0x7FFFFF00      // Largest 24-bit sample for peak detection (~24-bit max)

// --- Filter coefficients (Q8.24) ---
// Tuned for guitar, approximate center frequencies
#define BASS_A_Q24      (0x0003FD65) // 120 Hz
#define MID_A_Q24       (0x0013563F) // 600 Hz
#define TREBLE_A_Q24    (0x00579B7C) // 3.2 kHz

// Global HPF and LPF
#define HPF_A_Q24       (0x0002FF8C) // 90  Hz
#define LPF_A_Q24       (0x0092ACAE) // 6.5 kHz

// --- equalizer parameters in Q8.24 ---
static int32_t eq_gain          = 0x01000000;
static int32_t eq_volume        = 0x01000000;
static int32_t eq_low_gain_q24  = 0x01000000;
static int32_t eq_mid_gain_q24  = 0x01000000;
static int32_t eq_mid_a_q24     = MID_A_Q24;
static int32_t eq_high_gain_q24 = 0x01000000;
static int32_t eq_lpf_a_q24     = LPF_A_Q24;

static inline int32_t apply_1pole_iir(int32_t x, int32_t* state, int32_t a_q24) {
    int32_t diff = x - *state;
    *state += (int32_t)(((int64_t)diff * a_q24) >> 24);
    return *state;
}

static inline int32_t apply_1pole_hpf(int32_t x, int32_t* state, int32_t a_q24) {
    int32_t prev = *state;
    int32_t diff = x - prev;
    *state += (int32_t)(((int64_t)diff * a_q24) >> 24);
    return x - *state;
}

// Clamp 32-bit value to 24-bit value 
static inline __attribute__((always_inline)) int32_t clamp24(int32_t x) {
    if (x > PEAK_MAX) x = PEAK_MAX;
    if (x < PEAK_MIN) x = PEAK_MIN;
    return (int32_t)x;
}

// --- Per-channel equalizer processing ---
static inline __attribute__((always_inline)) int32_t process_eq_channel(
    int32_t s,
    int32_t *low_state,
    int32_t *mid_lp_state,
    int32_t *mid_hp_state,
    int32_t *high_state,
    int32_t *lpf_state,
    int32_t *hpf_state
) {
    s = (int32_t)(((int64_t)s * eq_gain) >> 24);

    // HPF before clipping to reduce rumble
    s = apply_1pole_hpf(s, hpf_state, HPF_A_Q24);   // Global HPF

    // LPF 
    s = apply_1pole_iir(s, lpf_state, eq_lpf_a_q24);   // Global LPF

    // Low-shelf
    int32_t low_out = apply_1pole_iir(s, low_state, BASS_A_Q24); // Global BASS
    low_out = (int32_t)(((int64_t)low_out * eq_low_gain_q24) >> 24);

    // Mid band-pass
    int32_t mid_band = apply_1pole_iir(
        apply_1pole_hpf(s, mid_hp_state, eq_mid_a_q24),
        mid_lp_state, eq_mid_a_q24
    );
    int32_t mid_out = (int32_t)(((int64_t)mid_band * eq_mid_gain_q24) >> 24);

    // High-shelf filter
    int32_t high_out = s - apply_1pole_iir(s, high_state, TREBLE_A_Q24); // Global TREB
    high_out = (int32_t)(((int64_t)high_out * eq_high_gain_q24) >> 24);

    // Mix Tonestack
    int32_t y = low_out + mid_out + high_out;
    y = (int32_t)(((int64_t)y * eq_volume) >> 24);

    return clamp24(y);
}