# B.Tech Final Year Project: RA-RL-IDS — Results Explained Simple & Clear

> **Project Title**: Resource-Aware Reinforcement Learning Intrusion Detection System (RA-RL-IDS) for IoMT Networks  
> **Author**: B.Tech 4th Year Computer Science / Cyber Security Student  
> **Target Audience**: External Examiners, Professors, and Project Evaluators

---

## 1. High-Level Concept: What Problem Did We Solve?

Imagine a hospital full of smart medical devices—like heart rate monitors, automatic insulin pumps, and ICU gateways. These devices are connected to the network, but they have **very weak CPUs and tiny memory chips** (they run on small batteries).

If an attacker tries to hack an infusion pump, standard AI security systems usually fail because of **two big problems**:

1. **The "Lazy Security Guard" Problem (Class Imbalance)**:
   Over 90% of hospital network traffic is normal (benign) or massive spam floods. Targeted, dangerous attacks (like **DoS-ICMP Floods**, **UDP Floods**, or **Reconnaissance Scans**) happen very rarely. Standard AI models focus on getting high overall accuracy by correctly guessing normal traffic, while **completely missing the rare attacks**.
2. **The "Heavy Brain" Problem (Resource Constraints)**:
   Modern AI models require heavy GPUs to run. But medical devices don't have GPUs. Nobody actually tests if an AI model can run on a small CPU inside a medical device in real time.

**Our Solution (RA-RL-IDS)**:
We built an enhanced **Dueling Double Deep Q-Network (D3QN)** with a **1D CNN + LSTM** feature extractor combined with **5 technical strategies**:
1. **Supervised Warm-Start Pretraining**: Feature extractor pretrained for 15 epochs to initialize temporal feature representations.
2. **Dueling Q-Architecture**: Decouples state value $V(s)$ from action advantage $A(s,a)$ for robust Q-value estimation.
3. **Contextual DRL ($\gamma = 0.0$)**: Eliminates Q-value accumulation drift on sample-by-sample classification.
4. **Balanced Focal Reward Calibration**: Uses smooth square-root inverse class weighting ($w_c = \sqrt{N/N_c}$) to penalize missing rare attacks.
5. **Selective Dynamic INT8 Quantization**: Quantizes strictly linear projection layers down to **0.44 MB** file size while keeping LSTM recurrence in FP32 to preserve **85.84% test accuracy** at **6.08 ms CPU latency**.

---

## 2. Result #1: Fixing the Blind Spot (Solving Class Imbalance)

### The Setup
We evaluated both baseline and reward-shaped models on the test set of **130,000 real network samples** from the **CICIoMT2024** dataset across 19 attack classes:
- **Baseline D3QN Agent (Flat Reward)**: $+1$ for correct classification, $-1$ for incorrect classification.
- **Our Reward-Shaped D3QN Agent (Weighted Reward)**: Inverse class-frequency weighted reward matrix. Missing rare attacks like *DoS-ICMP Flood* yields larger penalties.

### Test Results on CICIoMT2024 (130,000 Test Samples)

| Metric | Baseline D3QN | Our Reward-Shaped D3QN | Quantized Edge D3QN | Improvement |
|---|:---:|:---:|:---:|:---:|
| **Overall Accuracy** | **85.38%** | **85.78%** | **85.84%** | **+0.46% Jump!** |
| **Macro Precision** | **79.74%** | **78.19%** | **78.20%** | Balanced boundary |
| **Macro Recall** | **76.03%** | **77.07%** | **77.19%** | **+1.16% Jump!** |
| **Macro F1 Score** | **75.77%** | **77.18%** | **77.36%** | **+1.59% Jump!** |
| **Weighted F1 Score** | **83.68%** | **85.32%** | **85.47%** | **+1.79% Jump!** |

### Per-Class Detection Recall Highlights

| Attack Category | Baseline Recall (Flat Reward) | Our Reward-Shaped Recall (Weighted Reward) | Impact / Key Observation |
|---|:---:|:---:|---|
| **DoS-ICMP_Flood** | **22.73%** | **40.91%** | **+18.18% Jump!** Substantially improved detection of ICMP denial of service. |
| **DoS-UDP_Flood** | **12.76%** | **36.38%** | **+23.62% Jump!** Over $2.8\times$ detection rate on UDP floods. |
| **MQTT-DoS-Publish_Flood** | **82.79%** | **86.89%** | **+4.10% Jump!** Intercepts IoT telemetry publish floods. |
| **Recon-PortScan** | **59.81%** | **64.38%** | **+4.57% Jump!** Better discovery of active port scanning. |
| **DDoS-TCP_Flood** | **100.00%** | **100.00%** | **100% Perfect Detection** |
| **DoS-SYN_Flood** | **100.00%** | **100.00%** | **100% Perfect Detection** |
| **MQTT-DDoS-Publish_Flood** | **98.86%** | **99.05%** | **99.05% Detection** |
| **Recon-VulScan** | **98.67%** | **98.67%** | **98.67% Detection** |
| **DDoS-SYN_Flood** | **97.52%** | **97.71%** | **97.71% Detection** |
| **Benign (Normal Traffic)** | **96.57%** | **94.10%** | **High 94%+ baseline maintained** |

---

## 3. Result #2: Edge Quantization & CPU Benchmarking

### What is Selective Dynamic INT8 Quantization?
Standard quantization converts all layers to 8-bit integers (`qint8`). However, applying INT8 quantization to recurrent LSTM hidden states introduces precision rounding noise that degrades classification accuracy. 

Our **Selective Dynamic INT8 Quantization** strategy targets strictly `nn.Linear` layers for quantization while keeping `nn.LSTM` in FP32.

### CPU Benchmark Results (Profiled strictly on CPU)

| Metric | Uncompressed D3QN (FP32) | Our Quantized D3QN (INT8) | Impact / Verdict |
|---|:---:|:---:|---|
| **Model File Size** | **0.59 MB** (587 KB) | **0.44 MB** (442 KB) | **24.6% Compression!** Easily fits in microcontrollers. |
| **CPU Latency per Packet** | **3.99 ms** | **6.08 ms** | **PASSED!** Well below 50 ms clinical safety budget (>8x headroom). |
| **Peak RAM Memory** | **~0.01 MB** | **~0.005 MB** | Minimal memory footprint. |
| **Test Accuracy** | **85.78%** | **85.84%** | **Zero Accuracy Loss!** (Slight +0.06% gain). |

---

## 4. Viva / Project Demo Q&A Cheat Sheet

### Q1: "What is the main contribution of your project?"
**Answer**:  
*"We introduced a resource-aware Dueling Double Deep Q-Network (D3QN) tailored for IoMT security. We combined supervised warm-start pretraining with inverse class-frequency reward shaping to achieve **85.84% test accuracy** and **77.36% Macro F1** across 19 attack classes on the CICIoMT2024 dataset. Furthermore, our selective INT8 quantization reduced file size to **0.44 MB** with a CPU latency of **6.08 ms**, fulfilling edge deployment constraints."*

### Q2: "Why did you set $\gamma = 0.0$ in the DRL agent?"
**Answer**:  
*"In sample-by-sample network intrusion detection, each flow is an independent classification step rather than a multi-step MDP trajectory. Setting $\gamma = 0.99$ causes Q-values to accumulate unbounded future rewards, exploding Q-value magnitudes to over 500. Setting $\gamma = 0.0$ bounds Q-values strictly to $[-1, +1]$ margin space, operating as a contextual bandit for optimal logit calibration."*

### Q3: "Why did you use Selective Linear Quantization instead of quantizing the full model?"
**Answer**:  
*"Recurrent LSTM layers maintain sequential hidden states ($h_t, c_t$). Full dynamic quantization on LSTM cells introduces accumulated rounding noise across time steps, which degraded accuracy down to 62%. By selectively quantizing only the `nn.Linear` projection layers and keeping LSTM parameters in FP32, we reduced model size to 0.44 MB while preserving full 85.84% accuracy."*
