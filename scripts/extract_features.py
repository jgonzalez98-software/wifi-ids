"""
extract_features.py — Extract AWID3-compatible features from a .pcap file.

Uses tshark to read the pcap and export exactly the 107 numeric fields the
trained model expects. Missing fields (e.g. SSH fields on non-SSH packets) are
left blank and will be filled by the model's imputer at inference time.

Usage:
    python scripts/extract_features.py --pcap data/captures/capture.pcap \
                                        --out  data/captures/features.csv

    # Label as normal traffic (class 0) for retraining:
    python scripts/extract_features.py --pcap data/captures/capture.pcap \
                                        --out  data/captures/features.csv \
                                        --label normal
"""

import argparse
import subprocess
import sys
import shutil
from pathlib import Path

import pandas as pd
import numpy as np

# Exact 107 features the model was trained on (from artifacts/models/feature_columns.joblib)
FEATURE_COLS = [
    "frame.encap_type", "frame.len", "frame.number", "frame.time_delta",
    "frame.time_delta_displayed", "frame.time_epoch", "frame.time_relative",
    "radiotap.channel.flags.cck", "radiotap.channel.flags.ofdm",
    "radiotap.channel.freq", "radiotap.datarate", "radiotap.length",
    "radiotap.mactime", "radiotap.timestamp.ts", "radiotap.vendor_oui",
    "wlan.duration", "wlan.fc.frag", "wlan.fc.order", "wlan.fc.moredata",
    "wlan.fc.protected", "wlan.fc.pwrmgt", "wlan.fc.type", "wlan.fc.retry",
    "wlan.fc.subtype", "wlan.fcs.bad_checksum", "wlan.fixed.beacon",
    "wlan.fixed.capabilities.ess", "wlan.fixed.capabilities.ibss",
    "wlan.fixed.reason_code", "wlan.fixed.timestamp", "wlan_radio.duration",
    "wlan.seq", "wlan_radio.channel", "wlan_radio.data_rate",
    "wlan_radio.end_tsf", "wlan_radio.frequency", "wlan_radio.signal_dbm",
    "wlan_radio.start_tsf", "wlan_radio.phy", "wlan_radio.timestamp",
    "wlan.rsn.capabilities.mfpc", "wlan_rsna_eapol.keydes.msgnr",
    "wlan_rsna_eapol.keydes.data_len", "wlan_rsna_eapol.keydes.key_info.key_mic",
    "eapol.keydes.key_len", "eapol.keydes.replay_counter", "eapol.len",
    "eapol.type", "arp.hw.type", "arp.hw.size", "arp.proto.size", "arp.opcode",
    "icmpv6.mldr.nb_mcast_records", "icmpv6.ni.nonce",
    "tcp.analysis.reused_ports", "smb.access.generic_execute",
    "smb.access.generic_read", "smb.access.generic_write", "smb.flags.notify",
    "smb.flags.response", "smb.flags2.nt_error", "smb.flags2.sec_sig",
    "smb.mid", "smb.nt_status", "smb.pid.high", "smb.tid", "smb2.acct",
    "smb2.domain", "smb2.write_length", "dhcp.client_id.duid_ll_hw_type",
    "dhcp.hw.addr_padding", "dhcp.option.vendor.bsdp.message_type",
    "dns.resp.len", "dns.retransmit_request", "dns.retransmit_response",
    "http.content_length", "http.next_request_in", "http.next_response_in",
    "http.request_in", "http.response_in", "http.time", "ssh.cookie",
    "ssh.compression_algorithms_client_to_server_length",
    "ssh.compression_algorithms_server_to_client_length", "ssh.direction",
    "ssh.dh_gex.max", "ssh.dh_gex.min", "ssh.dh_gex.nbits",
    "ssh.encryption_algorithms_client_to_server_length",
    "ssh.encryption_algorithms_server_to_client_length",
    "ssh.host_key.length", "ssh.host_key.type_length",
    "ssh.kex_algorithms_length",
    "ssh.mac_algorithms_client_to_server_length",
    "ssh.mac_algorithms_server_to_client_length", "ssh.message_code",
    "ssh.mpint_length", "ssh.packet_length", "ssh.packet_length_encrypted",
    "ssh.padding_length", "ssh.padding_string", "ssh.protocol",
    "ssh.server_host_key_algorithms_length", "tls.alert_message.desc",
    "tls.alert_message.level",
    "tls.compress_certificate.compressed_certificate_message.length",
    "tls.connection_id",
]


def check_tshark():
    if not shutil.which("tshark"):
        print("ERROR: tshark not found. Install with:", file=sys.stderr)
        print("  sudo apt install tshark", file=sys.stderr)
        sys.exit(1)


def extract_with_tshark(pcap_path: Path, tmp_csv: Path):
    """Run tshark to dump all 107 fields from the pcap into a CSV."""
    field_args = []
    for col in FEATURE_COLS:
        field_args += ["-e", col]

    cmd = [
        "tshark",
        "-r", str(pcap_path),
        "-T", "fields",
        *field_args,
        "-E", "header=y",
        "-E", "separator=,",
        "-E", "quote=d",
        "-E", "occurrence=f",   # take first occurrence when field repeats
    ]

    print(f"[*] Running tshark on {pcap_path} ...")
    with open(tmp_csv, "w") as f:
        result = subprocess.run(cmd, stdout=f, stderr=subprocess.PIPE, text=True)

    if result.returncode != 0:
        print("ERROR: tshark failed:", file=sys.stderr)
        print(result.stderr, file=sys.stderr)
        sys.exit(1)


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Convert to numeric, coerce non-numeric to NaN (matches training behavior)."""
    df = df.copy()

    # tshark outputs hex for some fields (e.g. "0x0001") — convert to int
    for col in df.columns:
        if df[col].dtype == object:
            # try hex first, then plain numeric
            def parse_val(v):
                if pd.isna(v) or str(v).strip() == "":
                    return np.nan
                s = str(v).strip()
                try:
                    return int(s, 16) if s.startswith("0x") or s.startswith("0X") else float(s)
                except ValueError:
                    return np.nan

            df[col] = df[col].apply(parse_val)

    return df


def add_labels(df: pd.DataFrame, label: str) -> pd.DataFrame:
    """Add attack_type and attack_id columns for use in retraining."""
    if label == "normal":
        df["attack_type"] = "0.Normal"
        df["attack_id"] = 0
    return df


def main():
    parser = argparse.ArgumentParser(description="Extract AWID3 features from a pcap file.")
    parser.add_argument("--pcap", required=True, help="Path to input .pcap or .pcapng file")
    parser.add_argument("--out", required=True, help="Path to output .csv file")
    parser.add_argument(
        "--label", choices=["normal", "none"], default="none",
        help="Add label columns: 'normal' sets attack_id=0 (for retraining). Default: none"
    )
    args = parser.parse_args()

    check_tshark()

    pcap_path = Path(args.pcap)
    out_path = Path(args.out)
    tmp_csv = out_path.with_suffix(".raw.csv")

    if not pcap_path.exists():
        print(f"ERROR: pcap file not found: {pcap_path}", file=sys.stderr)
        sys.exit(1)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Dump fields from pcap
    extract_with_tshark(pcap_path, tmp_csv)

    # 2. Load and clean
    print("[*] Loading and cleaning extracted fields...")
    df = pd.read_csv(tmp_csv, low_memory=False)
    print(f"    Raw rows: {len(df)}, columns: {len(df.columns)}")

    df = clean_dataframe(df)

    # 3. Optionally add labels
    if args.label != "none":
        df = add_labels(df, args.label)

    # 4. Save
    df.to_csv(out_path, index=False)
    tmp_csv.unlink(missing_ok=True)

    n_numeric = df.select_dtypes(include=[np.number]).shape[1]
    print(f"[*] Saved {len(df)} rows, {n_numeric} numeric features -> {out_path}")

    if args.label == "normal":
        print()
        print("Next step — add to training data:")
        print("  mkdir -p data/processed/CSV/0.Normal")
        print(f"  cp {out_path} data/processed/CSV/0.Normal/processed.csv")
        print("  python scripts/train_sample.py")
        print("  python -m src.models.train_rf")


if __name__ == "__main__":
    main()
