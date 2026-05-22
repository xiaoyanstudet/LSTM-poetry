#!/bin/zsh
set -e
cd "$(dirname "$0")"

if [ -x "/Users/yanguanlin/anaconda3/bin/python" ]; then
  PYTHON="/Users/yanguanlin/anaconda3/bin/python"
else
  PYTHON="python3"
fi

"$PYTHON" main.py

