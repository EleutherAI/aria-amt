#!/bin/bash

mkdir -p data models
[ -f data/audio.webm ] && rm ./data/audio.webm
[ -f data/audio.mp3 ] && rm ./data/audio.mp3

yt-dlp -x --audio-format mp3 --extract-audio --audio-quality 0 --no-playlist -o ./data/audio.mp3 --match-filter "duration <= 3h" $1
 
aria-amt transcribe \
    medium-double \
    models/piano-medium-double-1.0.safetensors \
    -load_path ./data/audio.mp3 \
    -save_dir ./data \
    -compile \
    -bs 1
