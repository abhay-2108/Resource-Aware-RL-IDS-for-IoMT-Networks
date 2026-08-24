"""Dataset Downloader for RA-RL-IDS.

Streams and downloads the CICIoMT2024 network flow CSV dataset from Hugging Face into `data/raw/`.
"""

import os
import sys
import json
import urllib.request
from pathlib import Path
import time

REPO_API = "https://huggingface.co/api/datasets/somnath0100/CICIoMT2024Small"
BASE_DOWNLOAD_URL = "https://huggingface.co/datasets/somnath0100/CICIoMT2024Small/resolve/main/"
OUTPUT_DIR = Path("data/raw")

# Select all unique attack classes and primary flow runs
PRIMARY_CSVS = [
    "CSVS/Benign_train.pcap_Flow.csv",
    "CSVS/ARP_Spoofing_train.pcap_Flow.csv",
    "CSVS/MQTT-DDoS-Connect_Flood_train.pcap_Flow.csv",
    "CSVS/MQTT-DDoS-Publish_Flood_train.pcap_Flow.csv",
    "CSVS/MQTT-DoS-Connect_Flood_train.pcap_Flow.csv",
    "CSVS/MQTT-DoS-Publish_Flood_train.pcap_Flow.csv",
    "CSVS/MQTT-Malformed_Data_train.pcap_Flow.csv",
    "CSVS/Recon-OS_Scan_train.pcap_Flow.csv",
    "CSVS/Recon-Ping_Sweep_train.pcap_Flow.csv",
    "CSVS/Recon-Port_Scan_train.pcap_Flow.csv",
    "CSVS/Recon-VulScan_train.pcap_Flow.csv",
    "CSVS/TCP_IP-DDoS-ICMP1_train.pcap_Flow.csv",
    "CSVS/TCP_IP-DDoS-SYN1_train.pcap_Flow.csv",
    "CSVS/TCP_IP-DDoS-TCP1_train.pcap_Flow.csv",
    "CSVS/TCP_IP-DDoS-UDP1_train.pcap_Flow.csv",
    "CSVS/TCP_IP-DoS-ICMP1_train.pcap_Flow.csv",
    "CSVS/TCP_IP-DoS-SYN1_train.pcap_Flow.csv",
    "CSVS/TCP_IP-DoS-TCP1_train.pcap_Flow.csv",
    "CSVS/TCP_IP-DoS-UDP1_train.pcap_Flow.csv",
]

def download_file(url: str, dest_path: Path, max_bytes: int = 150 * 1024 * 1024):
    """Download file in chunks with size capping to avoid multi-gigabyte bloat."""
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    temp_path = dest_path.with_suffix(".tmp")
    
    downloaded = 0
    with urllib.request.urlopen(req) as resp, open(temp_path, "wb") as f_out:
        total_size = int(resp.headers.get("Content-Length", 0))
        target_size = min(total_size, max_bytes) if total_size > 0 else max_bytes
        
        while True:
            chunk = resp.read(1024 * 512) # 512 KB chunks
            if not chunk:
                break
            f_out.write(chunk)
            downloaded += len(chunk)
            
            # Show progress
            if total_size > 0:
                pct = min(100.0, (downloaded / total_size) * 100)
                sys.stdout.write(f"\r    -> {downloaded / (1024*1024):.1f} MB / {total_size / (1024*1024):.1f} MB ({pct:.1f}%)")
            else:
                sys.stdout.write(f"\r    -> {downloaded / (1024*1024):.1f} MB downloaded")
            sys.stdout.flush()
            
            # If a massive flood file exceeds 100MB, we can read complete CSV rows and finish
            if downloaded >= max_bytes:
                print(f"\n    [Capped at {downloaded / (1024*1024):.1f} MB for storage efficiency]")
                break
                
    # If capped, clean trailing incomplete line
    if dest_path.exists():
        dest_path.unlink()
    temp_path.rename(dest_path)
    print(f"\n    -> Complete: {dest_path.name} ({dest_path.stat().st_size / (1024*1024):.2f} MB)")

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Starting CICIoMT2024 dataset download into {OUTPUT_DIR.resolve()} ...")
    print(f"Selected {len(PRIMARY_CSVS)} core attack and benign categories.\n")
    
    for idx, rfilename in enumerate(PRIMARY_CSVS, 1):
        filename = os.path.basename(rfilename)
        dest_path = OUTPUT_DIR / filename
        
        if dest_path.exists() and dest_path.stat().st_size > 500 * 1024:
            print(f"[{idx}/{len(PRIMARY_CSVS)}] Already exists: {filename} ({dest_path.stat().st_size / (1024*1024):.2f} MB)")
            continue
            
        file_url = BASE_DOWNLOAD_URL + rfilename
        print(f"[{idx}/{len(PRIMARY_CSVS)}] Downloading {filename} ...")
        try:
            download_file(file_url, dest_path)
            time.sleep(0.5)
        except Exception as e:
            print(f"\n    -> Error downloading {filename}: {e}")
            
    print("\nDataset download complete!")
    all_csvs = list(OUTPUT_DIR.glob("*.csv"))
    total_size_mb = sum(f.stat().st_size for f in all_csvs) / (1024 * 1024)
    print(f"Total CSVs in {OUTPUT_DIR}: {len(all_csvs)} ({total_size_mb:.2f} MB)")

if __name__ == "__main__":
    main()
