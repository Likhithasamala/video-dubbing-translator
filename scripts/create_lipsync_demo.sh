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

ffmpeg -hide_banner -loglevel error -y \
  -i "$ORIGINAL" -i "$DUBBED" \
  -filter_complex "\
[0:v]fps=30,scale=360:640,drawbox=x=0:y=0:w=iw:h=44:color=black@0.65:t=fill,\
drawtext=fontfile='${FONT}':text='${LABEL_L}':fontcolor=white:fontsize=19:x=12:y=12[L];\
[1:v]fps=30,scale=360:640,subtitles='${SRT}':force_style='FontName=Arial\,FontSize=11\,MarginV=14',\
drawbox=x=0:y=0:w=iw:h=44:color=black@0.65:t=fill,\
drawtext=fontfile='${FONT}':text='${LABEL_R}':fontcolor=white:fontsize=19:x=12:y=12[R];\
[L][R]hstack=inputs=2:shortest=1[v]" \
  -map "[v]" -map 1:a -c:v libx264 -crf 18 -preset medium -pix_fmt yuv420p \
  -c:a aac -b:a 192k -movflags +faststart "$OUTPUT"

echo "Created $OUTPUT"
