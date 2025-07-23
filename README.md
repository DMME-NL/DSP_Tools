# DSP_Tools
Toolkits for designing DSP effects in C | Python

## First order fixed point tonestack design tool
As I generaly use RP2040 microcontrollers for DSP audio, there are not a lot of resources available.
Most of my projects focus on guitar. Therefore, this tool is created to design a quick and easy tonestack.

It consists of a HPF -> (Bass shelf + Mid band + Treble shelf) -> LPF.
All the corner frequencies are fully adjustable and the Python script let's you listen to it in real time.
The filter coefficients are calauclted in fixed point (Q8.24) format, to be used in C programs.

<img width="938" height="816" alt="image" src="https://github.com/user-attachments/assets/6d593099-0be8-4b5c-b30c-e5c89b4f801b" />
