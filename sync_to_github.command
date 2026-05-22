#!/bin/zsh
set -e
cd "$(dirname "$0")"

REMOTE_URL="https://github.com/xiaoyanstudet/LSTM-poetry.git"

if git remote get-url origin >/dev/null 2>&1; then
  git remote set-url origin "$REMOTE_URL"
else
  git remote add origin "$REMOTE_URL" 2>/dev/null || true
fi

echo "Current Git status:"
git status --short
echo

echo "Commit message:"
read message

if [ -z "$message" ]; then
  message="Update project"
fi

git add .

if git diff --cached --quiet; then
  echo "No changes to commit."
  exit 0
fi

git commit -m "$message"

if git remote get-url origin >/dev/null 2>&1; then
  git push -u origin main
else
  git push -u "$REMOTE_URL" main
fi
