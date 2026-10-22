#!/usr/bin/env bash
# Build a side-by-side manager demo: original video (left) vs dubbed + lip-synced (right),
# with panel labels and the dubbed subtitles burned in under the right panel.
# Usage: create_lipsync_demo.sh ORIGINAL DUBBED_LIPSYNC SUBTITLES OUTPUT [LABEL_LEFT] [LABEL_RIGHT]
set -euo pipefail
cd "$(dirname "$0")/.."

ORIGINAL="$1"
DUBBED="$2"
SRT="$3"
OUTPUT="$4"
LABEL_L="${5:-Original}"
LABEL_R="${6:-Dubbed + AI lip sync}"

mkdir -p "$(dirname "$OUTPUT")"

# An explicit fontfile avoids fontconfig, which is missing in many Windows FFmpeg builds.
FONT="C\\:/Windows/Fonts/arialbd.ttf"

# Fit each panel inside 640x640, keeping the source aspect (even dimensions for yuv420p).
read -r SRC_W SRC_H < <(ffprobe -v error -select_streams v:0 -show_entries stream=width,height \
  -of csv=p=0 "$ORIGINAL" | tr -d '\r' | tr ',' ' ')
if (( SRC_W >= SRC_H )); then
  PW=640; PH=$(( (SRC_H * 640 / SRC_W) / 2 * 2 ))
else
  PH=640; PW=$(( (SRC_W * 640 / SRC_H) / 2 * 2 ))
fi
FS=$(( PW / 19 ))           # label font size scales with panel width
BAR=$(( FS * 7 / 3 ))       # label bar height
SUBFS=$(( PW / 33 ))        # burned-subtitle size

ffmpeg -hide_banner -loglevel error -y \
  -i "$ORIGINAL" -i "$DUBBED" \
  -filter_complex "\
[0:v]fps=30,scale=${PW}:${PH},drawbox=x=0:y=0:w=iw:h=${BAR}:color=black@0.65:t=fill,\
drawtext=fontfile='${FONT}':text='${LABEL_L}':fontcolor=white:fontsize=${FS}:x=12:y=$((BAR / 4))[L];\
[1:v]fps=30,scale=${PW}:${PH},subtitles='${SRT}':force_style='FontName=Arial\,FontSize=${SUBFS}\,MarginV=14',\
drawbox=x=0:y=0:w=iw:h=${BAR}:color=black@0.65:t=fill,\
drawtext=fontfile='${FONT}':text='${LABEL_R}':fontcolor=white:fontsize=${FS}:x=12:y=$((BAR / 4))[R];\
[L][R]hstack=inputs=2:shortest=1[v]" \
  -map "[v]" -map 1:a -c:v libx264 -crf 18 -preset medium -pix_fmt yuv420p \
  -c:a aac -b:a 192k -movflags +faststart "$OUTPUT"

echo "Created $OUTPUT"
