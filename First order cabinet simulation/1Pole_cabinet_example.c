// Generated with LLM based on the implementation in Python
// Not tested in any way shape or form

#include <stdint.h>

#define Q24_ONE         0x01000000
#define PEAK_MAX        0x7FFFFF00
#define PEAK_MIN       -0x7FFFFF00

FilterStage filters[6] = {
    {1, 0, HPF,  Q24_ONE, 0,0, 0x00200000, 0},
    {1, 1, BPF,  Q24_ONE, 0,0, 0x00300000, 0x00600000},
    {1, 1, BPF,  Q24_ONE, 0,0, 0x01000000, 0x02000000},
    {1, 1, BPF,  Q24_ONE, 0,0, 0x04000000, 0x06000000},
    {1, 0, LPF,  Q24_ONE, 0,0, 0x08000000, 0},
    {1, 0, LPF,  Q24_ONE, 0,0, 0x0A000000, 0}
};


typedef enum { LPF, HPF, BPF, BSF } FilterType;

typedef struct {
    uint8_t enabled;
    uint8_t parallel;
    FilterType type;
    int32_t gain_q24;

    // State for filters (LPF/HPF and BPF/BSF need two)
    int32_t state1;
    int32_t state2;

    // Coefficients (Q8.24)
    int32_t a1_q24; // LPF/HPF: main coefficient or BPF/BSF low
    int32_t a2_q24; // BPF/BSF: high cutoff
} FilterStage;

static inline int32_t clamp24(int32_t x) {
    if (x > PEAK_MAX) x = PEAK_MAX;
    if (x < PEAK_MIN) x = PEAK_MIN;
    return x;
}

static inline int32_t apply_1pole_lpf(int32_t x, int32_t* state, int32_t a_q24) {
    int32_t diff = x - *state;
    *state += (int32_t)(((int64_t)diff * a_q24) >> 24);
    return *state;
}

static inline int32_t apply_1pole_hpf(int32_t x, int32_t* state, int32_t a_q24) {
    int32_t prev = *state;
    *state += (int32_t)(((int64_t)(x - prev) * a_q24) >> 24);
    return x - *state;
}

// Apply a single filter stage to a sample
static int32_t process_filter_stage(FilterStage* f, int32_t x) {
    if (!f->enabled) return x;

    int32_t y = 0;

    switch (f->type) {
        case LPF:
            y = apply_1pole_lpf(x, &f->state1, f->a1_q24);
            break;
        case HPF:
            y = apply_1pole_hpf(x, &f->state1, f->a1_q24);
            break;
        case BPF: {
            int32_t hp = apply_1pole_hpf(x, &f->state1, f->a1_q24);
            y = apply_1pole_lpf(hp, &f->state2, f->a2_q24);
            break;
        }
        case BSF: {
            int32_t lp = apply_1pole_lpf(x, &f->state1, f->a1_q24);
            int32_t hp = apply_1pole_hpf(x, &f->state2, f->a2_q24);
            y = lp + hp;
            break;
        }
    }

    // Apply gain
    return (int32_t)(((int64_t)y * f->gain_q24) >> 24);
}

#define MAX_STAGES 6

int32_t process_cabinet(int32_t x, FilterStage* stages) {
    int32_t group_sum = 0;
    int group_count = 0;

    for (int i = 0; i < MAX_STAGES; ++i) {
        FilterStage* f = &stages[i];
        if (!f->enabled) continue;

        int32_t y = process_filter_stage(f, x);

        if (f->parallel) {
            group_sum += y;
            group_count++;
        } else {
            if (group_count > 0) {
                x = group_sum / group_count;
                group_sum = 0;
                group_count = 0;
            }
            x = y;
        }

        if (i == MAX_STAGES - 1 && group_count > 0) {
            x = group_sum / group_count;
        }
    }

    return clamp24(x);
}
