#!/usr/bin/env bash
# Publish outlook.json to the single-commit `outlook` branch. Only the outlook workflow writes
# this branch, so a plain force-push is safe. The bot's `state` branch is never touched.
set -euo pipefail
[ -f outlook.json ] || { echo "nothing to publish"; exit 0; }
remote="${PUBLISH_REMOTE:-https://x-access-token:${GH_TOKEN:-}@github.com/${GITHUB_REPOSITORY:-}.git}"
pub=$(mktemp -d)
trap 'rm -rf "$pub"' EXIT
cp outlook.json "$pub/"
cd "$pub"
git init -q -b outlook
git config user.name "paper-bot"
git config user.email "paper-bot@users.noreply.github.com"
git add -A
git commit -q -m "outlook $(date -u +%FT%TZ)"
git push -q --force "$remote" outlook
echo "published outlook"
