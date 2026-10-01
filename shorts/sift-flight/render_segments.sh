#!/bin/bash
# Render the whole video in one-second segments (resumable: finished segments are kept), join them, then the
# soundtrack and the YouTube encode.   LOOK=sift_night bash render_segments.sh output/night
set -e
OUT=${1:-output/final}
cd "$(dirname "$0")/src"
FF=$(python -c "import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())")
mkdir -p "../$OUT"
: > "../$OUT/list.txt"
for k in $(seq 0 14); do
  seg="../$OUT/seg_$(printf %02d $k)"
  if [ ! -f "$seg/video.mp4" ]; then
    python main.py --out "$seg" --res 1440 --ss 2 --from $k --to $((k + 1)) --no-audio > "$seg.log" 2>&1
  fi
  echo "file '$(realpath $seg/video.mp4)'" >> "../$OUT/list.txt"
  echo "segment $k done"
done
"$FF" -y -v error -f concat -safe 0 -i "../$OUT/list.txt" -c copy "../$OUT/video.mp4"
python main.py --out "../$OUT" --res 1440 --encode-only
echo ALL DONE
