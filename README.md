# DSP_Tools
Toolkits for designing DSP effects in C | Python

These tools where created for quick and intuitive design of filters, tone stacks, or cabinets. 
Especially for guitar-focused DSP projects on the RP2040 microcontroller. 
Since fixed-point audio DSP resources are limited, this project helps bridge the gap.

These tools where created for quick and intuitive design of filters, tone stacks, or cabinets. 
Especially for guitar-focused DSP projects on the RP2040 microcontroller. 
Since fixed-point audio DSP resources are limited, this project helps bridge the gap.

## First-Order Fixed-Point Tone Stack Design Tool
Models a simple tone stack consisting of: HPF → (Bass Shelf + Mid Band + Treble Shelf) → LPF
Folder includes a snipet of C-code from a DSP project I have been workin on.
It should show the basics of implementing the filters in C, and get you started. 

<img width="625" height="544" alt="image" src="https://github.com/user-attachments/assets/6d593099-0be8-4b5c-b30c-e5c89b4f801b" />

## First-Order Fixed-Point Filter Design Tool
A simple tool for modeling a low-pass, high-pass, band-pass, or band-stop filter

<img width="630" height="335" alt="image" src="https://github.com/user-attachments/assets/12a0f998-0193-4585-a5f6-9f0090ec7772" />


## First-Order Fixed-Point Filter Design Tool
Combine three filters to replicate guitar speaker cabinets, or any other system for that matter.
It is not a very close approximation of a real speaker by any means, but it is very flexible for tone-shaping.

<img width="690" height="433" alt="image" src="https://github.com/user-attachments/assets/077ed706-765c-4d01-9e4c-dab9104cdfe1" />

## Note
The real time playback can crash the tool when using 'wrong' sample rate or device.
I had some good results enabling Realtek stereo-mix after making sure audio is present.
Windows WASAPI is not the best for low latency, so the BLOCK_SIZE needs to be set quite high.
Using a multi-channel audio interface also works, but I had to connect a cable from output 3-4 to the inputs.
Output 1-2 is then connected to speakers / headset to listen to the filtered PC sound.
It would be possible to modify the code for ASIO, but to keep it general WASAPI will do.
