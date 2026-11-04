#!/usr/bin/env sh
set -e
cd "$(dirname "$0")"
cargo build --release
cp target/release/mequa_launcher ../Mequa
echo "Built ../Mequa"
