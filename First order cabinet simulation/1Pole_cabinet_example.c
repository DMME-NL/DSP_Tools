#ifndef SPEAKER_SIM_H
#define SPEAKER_SIM_H

#define Q24_ONE     0x01000000
#define SAMPLE_RATE 48000
#define CAB_SIM_EFFECT_INDEX 12  // example index

#define NUM_POTS    6
#define MAX_EFFECTS 32
#define POT_MAX     4095    // Max pot value (12 bit)

extern int storedPotValue[MAX_EFFECTS][NUM_POTS];
extern int pot_value[NUM_POTS];

// ============================================================================
// === Audio Processing Functions =============================================
// ============================================================================

#define PEAK_MAX        0x7FFFFF00      // Largest 24-bit sample for peak detection (~24-bit max)
#define PEAK_MIN       -0x7FFFFF00      // Largest 24-bit sample for peak detection (~24-bit max)

// === Filter structs ===
typedef struct {
    int32_t a_q24;
    int32_t state_l, state_r;
} OnePole;

typedef struct {
    OnePole hpf, lpf;
    int32_t gain_q24;
    int32_t s1_l, s2_l;
    int32_t s1_r, s2_r;
} BPFPair;

// Apply a 1-pole IIR filter
static inline int32_t apply_1pole_lpf(int32_t x, int32_t* state, int32_t a_q24) {
    int32_t diff = x - *state;
    *state += (int32_t)(((int64_t)diff * a_q24) >> 24);
    return *state;
}

// Apply a 1-pole HPF filter
static inline int32_t apply_1pole_hpf(int32_t x, int32_t* state, int32_t a_q24) {
    int32_t prev = *state;
    int32_t diff = x - prev;
    *state += (int32_t)(((int64_t)diff * a_q24) >> 24);
    return x - *state;
}

// Band Pass filter
static inline int32_t apply_1pole_bpf(int32_t x, BPFPair* f, int ch) {
    int32_t* s1 = (ch == 0) ? &f->hpf.state_l : &f->hpf.state_r;
    int32_t* s2 = (ch == 0) ? &f->lpf.state_l : &f->lpf.state_r;

    int32_t hp = apply_1pole_hpf(x, s1, f->hpf.a_q24);
    int32_t bp = apply_1pole_lpf(hp, s2, f->lpf.a_q24);

    return qmul(bp, f->gain_q24);
}

// Band Stop filter
static inline int32_t apply_1pole_bsf(int32_t x, BPFPair* f, int ch) {
    int32_t* s1 = (ch == 0) ? &f->hpf.state_l : &f->hpf.state_r;
    int32_t* s2 = (ch == 0) ? &f->lpf.state_l : &f->lpf.state_r;

    int32_t hp = apply_1pole_hpf(x, s1, f->hpf.a_q24);
    int32_t bp = apply_1pole_lpf(hp, s2, f->lpf.a_q24);

    int32_t notch = x - bp;
    return qmul(notch, f->gain_q24);
}

// ============================================================================
// === Data conversion and math functions =====================================
// ============================================================================

static inline float fast_db_to_gain(float db) {
    // Approximate 10^(db/20)
    return powf(10.0f, db / 20.0f);
}

// --- Utility function: convert dB to linear (approx) ---
static inline int32_t db_to_q24(float db) {
    // 20*log10(gain) = db → gain = 10^(db/20)
    float lin = powf(10.0f, db / 20.0f);
    return (int32_t)(lin * (1 << 24));
}

static inline uint32_t float_to_q16(float x) {
    // Convert float to Q16
    return (uint32_t)(x * Q16_ONE);
}

static inline float q16_to_float(int32_t x) {
    return x / 65536.0f;
}

static inline int32_t float_to_q24(float x) {
    // Convert float to Q24
    return (int32_t)(x * (1 << 24));
}

static inline float q24_to_float(int32_t x) {
    return x / 16777216.0f;
}

static inline int32_t map_pot_to_q24(int32_t pot, int32_t min_q24, int32_t max_q24) {
    // Convert pot to Q24
    return min_q24 + ((int64_t)pot * (max_q24 - min_q24)) / POT_MAX;
}

static inline int32_t map_pot_to_int(int32_t pot, int32_t min_int, int32_t max_int) {
    // Convert pot to INT
    return min_int + ((int64_t)pot * (max_int - min_int)) / POT_MAX;
}

// Clamp 32-bit value to 24-bit value 
static inline __attribute__((always_inline)) int32_t clamp24(int32_t x) {
    if (x > PEAK_MAX) x = PEAK_MAX;
    if (x < PEAK_MIN) x = PEAK_MIN;
    return (int32_t)x;
}

// Approximate: 1 - exp(-6.28318 * x) using a rational function
int32_t fc_to_q24(uint32_t fc, uint32_t fs) {
    if (fc >= fs / 2)
        return 0xFFFFFF; // Q24_ONE

    double normalized = (double)fc / fs;
    double coeff = 2.0 * sin(M_PI * normalized);
    return (int32_t)(coeff * (1 << 24) + 0.5);
}

static inline int32_t qmul(int32_t a, int32_t b) {
    return (int32_t)(((int64_t)a * b) >> 24);
}


// === Filter instances ===
static OnePole hpf0, lpf4, lpf5;
static BPFPair bpf1, bpf2, bpf3;
static int32_t cab_output_gain_q24 = Q24_ONE;

static inline void set_bpf_cutoffs(BPFPair* f, int32_t fc, int32_t bw) {
    int32_t fc_low = fc - bw / 2;
    int32_t fc_high = fc + bw / 2;

    // Clamp within valid audio range
    if (fc_low < 20) fc_low = 20;
    if (fc_high > SAMPLE_RATE / 2) fc_high = SAMPLE_RATE / 2;

    f->hpf.a_q24 = fc_to_q24(fc_low, SAMPLE_RATE);
    f->lpf.a_q24 = fc_to_q24(fc_high, SAMPLE_RATE);
}

// === Processing ===
static inline int32_t process_speaker_channel(int32_t x, int ch) {
    // Stage 0: HPF
    int32_t* hpf_state = (ch == 0) ? &hpf0.state_l : &hpf0.state_r;
    int32_t y = apply_1pole_hpf(x, hpf_state, hpf0.a_q24);

    // Parallel group
    int32_t p1 = apply_1pole_bpf(x, &bpf1, ch);
    int32_t p2 = apply_1pole_bpf(x, &bpf2, ch);
    int32_t p3 = apply_1pole_bpf(x, &bpf3, ch);
    y += (p1 + p2 + p3) / 3;

    // LPF 5kHz with -2dB
    int32_t* lpf4_state = (ch == 0) ? &lpf4.state_l : &lpf4.state_r;
    y = apply_1pole_lpf(y, lpf4_state, lpf4.a_q24);

    // LPF 8kHz with +6dB
    int32_t* lpf5_state = (ch == 0) ? &lpf5.state_l : &lpf5.state_r;
    y = apply_1pole_lpf(y, lpf5_state, lpf5.a_q24);

    // Apply gain (6dB) to the output
    y = qmul(y, db_to_q24(6.0f));

    // Output gain (controlled by pot)
    return clamp24(qmul(y, cab_output_gain_q24));
}

static inline void speaker_sim_process_sample(int32_t* inout_l, int32_t* inout_r) {
    *inout_l = process_speaker_channel(*inout_l, 0);
    *inout_r = process_speaker_channel(*inout_r, 1);
}

void speaker_sim_process_block(int32_t* in_l, int32_t* in_r, size_t frames) {
    for (size_t i = 0; i < frames; ++i)
        speaker_sim_process_sample(&in_l[i], &in_r[i]);
}

// === Initialization ===
static inline void init_speaker_sim(void) {
    hpf0.a_q24 = fc_to_q24(80, SAMPLE_RATE);

    set_bpf_cutoffs(&bpf1, 120, 80);    // Fc = 120, BW = 80 → 80–160 Hz
    bpf1.gain_q24 = db_to_q24(5.0f);

    set_bpf_cutoffs(&bpf2, 600, 500);   // Fc = 600, BW = 500 → 375–825 Hz
    bpf2.gain_q24 = db_to_q24(-4.0f);

    set_bpf_cutoffs(&bpf3, 2500, 1200); // Fc = 2500, BW = 1200 → 1900–3100 Hz
    bpf3.gain_q24 = db_to_q24(6.0f);

    lpf4.a_q24 = fc_to_q24(5000, SAMPLE_RATE);
    lpf5.a_q24 = fc_to_q24(8000, SAMPLE_RATE);

    cab_output_gain_q24 = Q24_ONE;
}

static inline void load_speaker_sim_parms_from_memory(void) {
    int32_t pot;

    // === Pot 0: Low Cut HPF (30–200 Hz) ===
    pot = storedPotValue[CAB_SIM_EFFECT_INDEX][0];
    int32_t hpf_freq = map_pot_to_int(pot, 30, 200);  // Hz
    hpf0.a_q24 = fc_to_q24(hpf_freq, SAMPLE_RATE);

    // === Pot 1: Body Gain (–6 dB to +12 dB) ===
    pot = storedPotValue[CAB_SIM_EFFECT_INDEX][1];
    int32_t body_gain_q24 = map_pot_to_q24(pot, db_to_q24(-6.0f), db_to_q24(12.0f));
    bpf1.gain_q24 = body_gain_q24;

    // === Pot 2: Mid Scoop (–10 dB to 0 dB) ===
    pot = storedPotValue[CAB_SIM_EFFECT_INDEX][2];
    int32_t mid_dip_q24 = map_pot_to_q24(pot, db_to_q24(-10.0f), db_to_q24(0.0f));
    bpf2.gain_q24 = mid_dip_q24;

    // === Pot 3: Presence Gain (–6 dB to +12 dB) ===
    pot = storedPotValue[CAB_SIM_EFFECT_INDEX][3];
    int32_t pres_gain_q24 = map_pot_to_q24(pot, db_to_q24(-6.0f), db_to_q24(12.0f));
    bpf3.gain_q24 = pres_gain_q24;

    // === Pot 4: Air Freq (LPF5) – 3kHz to 10kHz ===
    pot = storedPotValue[CAB_SIM_EFFECT_INDEX][4];
    int32_t air_freq = map_pot_to_int(pot, 3000, 10000);
    lpf5.a_q24 = fc_to_q24(air_freq, SAMPLE_RATE);

    // === Pot 5: Output Volume (0.2x to 2.0x linear gain) ===
    pot = storedPotValue[CAB_SIM_EFFECT_INDEX][5];
    cab_output_gain_q24 = map_pot_to_q24(pot, float_to_q24(0.2f), float_to_q24(2.0f));
}

static inline void update_speaker_sim_params_from_pots(int changed_pot) {
    if (changed_pot < 0 || changed_pot >= 6) return;
    storedPotValue[CAB_SIM_EFFECT_INDEX][changed_pot] = pot_value[changed_pot];
    load_speaker_sim_parms_from_memory();
}


#endif // SPEAKER_SIM_H
