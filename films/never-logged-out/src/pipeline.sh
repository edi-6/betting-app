#!/bin/sh
# Everything after the code: safe to run again after an interruption (render and compose skip finished shots).
set -e
cd "$(dirname "$0")"
python film.py render
python film.py compose
[ -f ../output/film_audio.wav ] || { python film.py cues && python audio.py; }
python film.py srt
python thumbnail.py
python deliver.py web ../output/web
python deliver.py git
echo "PIPELINE DONE"
