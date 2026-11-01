#!/usr/bin/env bash
# Rebuild the viewer bundle (needs Node 18+). The built file is committed, so you only run this if you edit viewer.js.
set -e
cd "$(dirname "$0")"
[ -d node_modules ] || npm install --no-audit --no-fund
npx esbuild viewer.js --bundle --minify --format=iife --target=es2020 --outfile=../mequa/viewer/web/viewer.bundle.js
cp index.html ../mequa/viewer/web/index.html
