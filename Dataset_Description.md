# Dataset Description: CICIoMT2024 in RA-RL-IDS

This document provides a detailed description of the **CICIoMT2024** dataset utilized in the **RA-RL-IDS** project, covering its source, attack vectors, features, class imbalance characteristics, preprocessing pipeline, and synthetic generation design.

---

## 1. Dataset Provenance & Scope

*   **Dataset Name**: CICIoMT2024 (Attack vectors in healthcare devices - a multi-protocol dataset for assessing IoMT device security)
*   **Publisher**: Canadian Institute for Cybersecurity (CIC), University of New Brunswick (UNB)
*   **Year**: 2024
*   **Citation**: *Dadkhah, S. et al. (2024). "CICIoMT2024: Attack vectors in healthcare devices – a multi-protocol dataset for assessing IoMT device security," Internet of Things.* [DOI: 10.1016/j.iot.2024.101351](https://doi.org/10.1016/j.iot.2024.101351)
*   **Overview**: Modern healthcare systems rely on Internet of Medical Things (IoMT) devices (e.g., patient monitors, infusion pumps, BLE heart-rate monitors) operating under diverse network protocols. Prior security datasets (like KDD99 or CICIDS2017) represent standard IT networks and fail to capture specific IoMT protocols like MQTT or BLE. CICIoMT2024 addresses this gap by capturing real-world multi-protocol network traffic (Wi-Fi, Bluetooth/BLE, and MQTT) under 15 distinct attack scenarios and normal clinical operations.

---

## 2. Attack Taxonomy & Scenarios

The dataset features 15 distinct attack vectors, grouped into 4 high-level attack categories alongside normal traffic:

```
CICIoMT2024 Traffic
 ├── Benign (Normal Telemetry)
 ├── Denial of Service (DoS/DDoS)
 │    ├── ICMP Flood
 │    ├── TCP Flood
 │    ├── UDP Flood
 │    ├── SYN Flood
 │    ├── PSHACK Flood
 │    └── Synonymous IP Flood
 ├── Reconnaissance (Recon)
 │    ├── Host Discovery
 │    ├── Port Scan
 │    └── OS Scan
 └── Spoofing & Exploitation
      ├── ARP Spoofing
      ├── DNS Spoofing
      └── MQTT Publish Exploitation
```

### A. Benign Traffic
Normal operations of medical devices in a simulated smart hospital. This includes Wi-Fi traffic from monitors transmitting vital signs, MQTT clients publishing sensor readings to local brokers, and Bluetooth devices transmitting telemetry.

### B. Denial of Service (DoS & DDoS)
Attacks designed to flood the clinical network or device interface, exhausting bandwidth, memory, or CPU cycles. In medical settings, a DoS attack on an infusion pump or telemetry monitor can prevent doctors from receiving critical alerts or adjusting drug dosage.
*   **ICMP Flood**: Flooding the target with ICMP Echo Request packets (pings).
*   **TCP/UDP Flood**: Flooding random ports on the target with TCP/UDP packets, forcing it to check for listening applications.
*   **SYN Flood**: Exploiting the TCP three-way handshake by sending multiple SYN packets with spoofed source IPs and leaving the connections half-open.
*   **PSHACK Flood**: TCP flood using the PUSH and ACK flags to force the receiving device to immediately process the packet and empty its buffer.
*   **Synonymous IP Flood**: Sending flood packets where the source IP address is spoofed to match the target's IP, creating routing loops and resource exhaustion.

### C. Reconnaissance (Recon)
Pre-attack phases where adversaries probe the network to map out assets and discover vulnerabilities.
*   **Host Discovery**: Ping sweeps or ARP scans to identify active medical devices.
*   **Port Scan**: Scanning active ports (TCP/UDP) to identify what services (e.g. HTTP, MQTT, SSH) are running on the medical device.
*   **OS Scan**: Analyzing TCP/IP stack fingerprints to identify the exact operating system and version running on the IoMT hardware.

### D. Spoofing & Protocol Exploitation
Man-in-the-Middle (MitM) attacks and protocol exploits to hijack or spoof clinical commands.
*   **ARP Spoofing**: Sending spoofed ARP (Address Resolution Protocol) messages to associate the attacker's MAC address with the IP address of a legitimate medical device or gateway.
*   **DNS Spoofing**: Altering DNS server resolution records to redirect traffic from medical devices to malicious destinations.
*   **MQTT Publish Exploitation**: Intercepting or publishing unauthorized commands/telemetry to the MQTT broker, potentially tampering with dosage configurations.

---

## 3. Network Flow Features

Each record in the dataset represents a processed network flow containing **46 features**. These are divided into major categories:

1.  **Temporal & Flow Rate Features**:
    *   `flow_duration`: Total active time of the network connection.
    *   `Rate`: Packet transmission rate per second.
    *   `Srate`/`Drate`: Source-to-destination and destination-to-source packet rates.
    *   `IAT`: Inter-arrival time between consecutive packets.
2.  **Packet Header & Flag Count Features**:
    *   `Header_Length`: Total length of headers (IP, TCP, UDP).
    *   `Protocol Type`: Numeric representation of the protocol (e.g. TCP=6, UDP=17).
    *   `fin_flag_number`, `syn_flag_number`, `rst_flag_number`, `psh_flag_number`, `ack_flag_number`, `ece_flag_number`, `cwr_flag_number`: Binary flags indicating packet transmission control states.
    *   `syn_count`, `fin_count`, `ack_count`, `urg_count`, `rst_count`: Cumulative flag counters.
3.  **Application Protocol Identifiers**:
    *   `HTTP`, `HTTPS`, `DNS`, `Telnet`, `SMTP`, `SSH`, `IRC`, `TCP`, `UDP`, `DHCP`, `ARP`, `ICMP`, `IPv`, `LLC`: Binary flags representing protocol signatures detected in the flow payload.
4.  **Statistical Feature Aggregates**:
    *   `Tot sum`, `Min`, `Max`, `AVG`, `Std`, `Variance`: Statistics of packet sizes within the flow.
    *   `Magnitue`, `Radius`, `Covariance`, `Weight`: Clustering and density metrics of packets.

---

## 4. Class Distribution & Imbalance (Table 1)

The training partition contains a total of **49,000 samples** stratified into 16 categories. Normal benign traffic forms the majority, while highly critical exploits represent the minority. The inverse class-frequency weight $w_c$ is calculated to penalize minority classification errors harder:

| Class Index | Class Name | Train Count | Val Count | Test Count | Total Count | Class Weight (Normalized) | Rarity Level |
|:---:|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **0** | Benign | 10,500 | 2,250 | 2,250 | 15,000 | 0.1023 | Majority Class |
| **1** | DDoS-ICMP_Flood | 3,500 | 750 | 750 | 5,000 | 0.3070 | Low Imbalance |
| **2** | DDoS-PSHACK_Flood | 2,450 | 525 | 525 | 3,500 | 0.4385 | Low Imbalance |
| **3** | DDoS-SYN_Flood | 2,100 | 450 | 450 | 3,000 | 0.5116 | Moderate Imbalance |
| **4** | DDoS-SynonymousIP_Flood | 1,400 | 300 | 300 | 2,000 | 0.7674 | Moderate Imbalance |
| **5** | DDoS-TCP_Flood | 3,150 | 675 | 675 | 4,500 | 0.3411 | Low Imbalance |
| **6** | DDoS-UDP_Flood | 2,800 | 600 | 600 | 4,000 | 0.3837 | Low Imbalance |
| **7** | DoS-SYN_Flood | 1,050 | 225 | 225 | 1,500 | 1.0232 | High Imbalance |
| **8** | DoS-TCP_Flood | 1,750 | 375 | 375 | 2,500 | 0.6139 | Moderate Imbalance |
| **9** | DoS-UDP_Flood | 1,400 | 300 | 300 | 2,000 | 0.7674 | Moderate Imbalance |
| **10** | MQTT-Publish | 700 | 150 | 150 | 1,000 | 1.5348 | High Imbalance |
| **11** | Recon-HostDiscovery | 1,050 | 225 | 225 | 1,500 | 1.0232 | High Imbalance |
| **12** | Recon-OSScan | 525 | 112 | 113 | 750 | 2.0464 | High Imbalance |
| **13** | Recon-PortScan | 1,050 | 225 | 225 | 1,500 | 1.0232 | High Imbalance |
| **14** | Spoofing-ARP | 525 | 113 | 112 | 750 | 2.0464 | High Imbalance |
| **15** | Spoofing-DNS | 350 | 75 | 75 | 500 | 3.0697 | **Minority Class** |

---

## 5. Preprocessing & Feature Selection

Raw packet flow records must be preprocessed before being input to the CNN-LSTM neural network.

```
Raw CSV Datasets
 └── Median Imputation (NaN/Inf cleaning)
      └── StandardScaler Z-Score Scaling
           └── Mutual Information (MI) Scoring
                └── Top 25 Selected Feature Observation State
```

1.  **Cleaning**: Infinite values (`inf`, `-inf`) are replaced with `NaN`. All `NaN` cells are filled using the column's training median. Columns with zero variance (constant features across all samples) are dropped.
2.  **Scaling**: Features are scaled to have $\mu=0$ and $\sigma=1$ using standard scaling:
    $$z = \frac{x - \mu}{\sigma}$$
3.  **Feature Selection**: Rather than feeding all 46 raw features, we calculate the non-linear relationship between features and the target label. The top 25 features showing the highest mutual information are selected, reducing input dimensionality and computational overhead for edge systems. The selected features include `Srate`, `Rate`, and `Std`.

---

## 6. Synthetic Data Generator Design

To ensure the codebase runs out-of-the-box when raw CSVs are absent from `data/raw/`, `data_pipeline.py` features a **High-Fidelity Tabular Generator** that models the statistical signatures of the CICIoMT2024 dataset:
-   **Class Imbalance Preservation**: Generates the exact 16-class distribution and quantities.
-   **Class-Specific Feature Shifts**:
    *   *DDoS Classes*: Generated with elevated `Rate`, `Srate`, and packet size variations (`Std`).
    *   *DoS Classes*: Generated with elevated `Rate` and TCP `syn_flag_number` signals.
    *   *Recon Classes*: Generated with shortened `flow_duration` and high packet inter-arrival time (`IAT`) variations.
    *   *MQTT Classes*: Generated with elevated `Protocol Type` (identifying TCP/MQTT brokers) and high packet counts.
    *   *Spoofing Classes*: Elevated flag features in ARP and DNS columns.
-   **Random Noise Modeling**: Adds a standard Gaussian distribution ($\mathcal{N}(0, 1)$) multiplied by class-specific coefficients to simulate realistic network jitter.
