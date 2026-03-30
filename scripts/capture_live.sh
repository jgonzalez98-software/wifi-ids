#!/usr/bin/env bash
# capture_live.sh — Put a WiFi adapter in monitor mode and capture 802.11 traffic.
# Usage:
#   sudo bash scripts/capture_live.sh [interface] [duration_seconds] [output_pcap]
#
# Defaults:
#   interface      = wlan0
#   duration       = 60 seconds
#   output_pcap    = data/captures/capture_$(date).pcap
#
# After capture, run:
#   python scripts/extract_features.py --pcap <file> --out data/captures/features.csv

set -euo pipefail

IFACE="${1:-wlan0}"
DURATION="${2:-60}"
OUT_DIR="data/captures"
OUT_FILE="${3:-${OUT_DIR}/capture_$(date +%Y%m%d_%H%M%S).pcap}"

# ── checks ────────────────────────────────────────────────────────────────────
if [[ $EUID -ne 0 ]]; then
    echo "ERROR: This script must be run as root (sudo)." >&2
    exit 1
fi

if ! command -v tshark &>/dev/null; then
    echo "ERROR: tshark not found. Install with:"
    echo "  sudo apt install tshark"
    exit 1
fi

if ! ip link show "$IFACE" &>/dev/null; then
    echo "ERROR: Interface '$IFACE' not found."
    echo "Available interfaces:"
    ip link show | awk -F': ' '/^[0-9]+:/{print "  "$2}'
    exit 1
fi

mkdir -p "$OUT_DIR"

# ── monitor mode ──────────────────────────────────────────────────────────────
echo "[*] Bringing $IFACE down..."
ip link set "$IFACE" down

echo "[*] Switching $IFACE to monitor mode..."
iw dev "$IFACE" set type monitor

echo "[*] Bringing $IFACE up..."
ip link set "$IFACE" up

MONITOR_MODE=$(iw dev "$IFACE" info 2>/dev/null | awk '/type/{print $2}')
if [[ "$MONITOR_MODE" != "monitor" ]]; then
    echo "ERROR: Could not enable monitor mode on $IFACE (got: $MONITOR_MODE)."
    echo "Try using airmon-ng instead:"
    echo "  sudo airmon-ng start $IFACE"
    exit 1
fi

echo "[*] Monitor mode enabled on $IFACE."

# ── capture ───────────────────────────────────────────────────────────────────
echo "[*] Capturing for ${DURATION}s -> $OUT_FILE"
echo "    Press Ctrl+C to stop early."

cleanup() {
    echo ""
    echo "[*] Stopping capture..."
    # restore managed mode
    ip link set "$IFACE" down
    iw dev "$IFACE" set type managed
    ip link set "$IFACE" up
    echo "[*] $IFACE restored to managed mode."
    echo "[*] Capture saved to: $OUT_FILE"
    echo ""
    echo "Next step — extract features:"
    echo "  python scripts/extract_features.py --pcap $OUT_FILE --out data/captures/features.csv"
}
trap cleanup EXIT

tshark -i "$IFACE" \
    -w "$OUT_FILE" \
    -a duration:"$DURATION" \
    -q \
    2>&1 | grep -v "^Capturing on"

echo "[*] Done."
