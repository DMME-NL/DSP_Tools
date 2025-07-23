# DSP_Tools
Toolkits for designing DSP effects in C | Python

## First-Order Fixed-Point Tone Stack Design Tool
This tool was created for quick and intuitive design of tone stacks, especially for guitar-focused DSP projects on the RP2040 microcontroller. 
Since fixed-point audio DSP resources are limited, this project helps bridge the gap.

It models a simple tone stack consisting of:
HPF → (Bass Shelf + Mid Band + Treble Shelf) → LPF

All corner frequencies are fully adjustable via sliders, and the Python script allows you to visualize the frequency response and listen to the effect in real time. 
Filter coefficients are calculated in Q8.24 fixed-point format, ready to be used in embedded C applications.

<img width="938" height="816" alt="image" src="https://github.com/user-attachments/assets/6d593099-0be8-4b5c-b30c-e5c89b4f801b" />
