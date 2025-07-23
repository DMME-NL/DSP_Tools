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

## Note
The real time playback can crash the tool when using 'wrong' sample rate or device.
I had some good results enabling Realtek stereo-mix after making sure audio is present.
For some reason the sample rate needs to be quite high - expect some delay.
Using a multi-channel audio interface also works, but I had to connect a cable from output 3-4 to the inputs.
Output 1-2 is then connected to speakers / headset to listen to the filtered PC sound.