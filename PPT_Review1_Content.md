# 7-Slide Project Proposal (Review 1) — RA-RL-IDS

> **Project Title**: RA-RL-IDS: Resource-Aware Reinforcement Learning Intrusion Detection System for IoMT Networks  
> **Purpose**: Slide-by-slide detailed content designed for AI Slide Generators (e.g., Gamma, Canva, SlidesGPT, Pitch) and Review 1 Panel Presentation.

---

## Slide 1: Title Slide (Project Proposal & Introduction)

*   **Slide Title**: RA-RL-IDS: Resource-Aware Reinforcement Learning Intrusion Detection System for IoMT Networks
*   **Subtitle**: Solving Class Imbalance and Edge Deployability Gaps in Medical IoT Security
*   **Presenter Details**: 
    *   Student Name & Roll Number
    *   Department of Computer Science & Engineering / Cyber Security
    *   Project Advisor / Guide Name
    *   Date & Academic Session

### Speaker Notes (What to say out loud):
> *"Good morning respected members of the panel and my project guide. Today, I am proposing my B.Tech final year project titled 'RA-RL-IDS'—a Resource-Aware Reinforcement Learning Intrusion Detection System tailored specifically for Internet of Medical Things (IoMT) networks."*

---

## Slide 2: Motivation & Background (Why This Project?)

*   **Slide Header**: The Critical Need for Dedicated IoMT Security
*   **Key Points**:
    *   **Rise of Connected Healthcare**: Modern hospitals rely on network-connected medical devices—infusion pumps, ICU monitors, cardiac telemetry, and smart gateways.
    *   **High-Stakes Target**: Cyberattacks on healthcare IoMT directly jeopardize patient lives, data privacy, and hospital continuity.
    *   **Severe Resource Limits**: IoMT hardware operates under constrained CPUs, limited RAM, and battery power, making traditional bulky firewalls unfeasible.
    *   **Diverse Network Protocols**: Traffic spans Wi-Fi, Bluetooth/BLE, and lightweight protocols like MQTT, creating complex attack surfaces.

### Speaker Notes (What to say out loud):
> *"Medical devices are no longer standalone hardware; they are connected to hospital networks via Wi-Fi and MQTT protocols. While this enables real-time patient monitoring, it exposes life-critical devices to cyberattacks. However, because these devices run on tiny microcontrollers with minimal RAM, standard enterprise security software cannot be installed on them."*

---

## Slide 3: Problem Statement (Failures of Existing Solutions)

*   **Slide Header**: Limitations in Current DRL Intrusion Detection Literature
*   **Key Points**:
    *   **Gap A: The Deployability Gap (Hidden Hardware Costs)**:
        *   Existing DRL models are evaluated exclusively on high-end GPUs.
        *   Literature reports zero resource-cost profiling (RAM, CPU latency, storage size), leaving real-world edge deployment unverified.
    *   **Gap B: The Class Imbalance Gap (The Blind Spot)**:
        *   IoMT traffic is heavily imbalanced (~90%+ normal or DDoS flood traffic vs. <1% stealthy targeted attacks).
        *   Standard AI models optimize for overall dataset accuracy, masking near-zero detection rates on critical rare attacks (e.g., DNS Spoofing, OS Scanning).

### Speaker Notes (What to say out loud):
> *"When we surveyed recent DRL-IDS research papers, we identified two major flaws. First, researchers claim their models work for 'edge devices' but only test them on powerful GPUs without measuring CPU latency or RAM usage. Second, models report 99% overall accuracy, but when we look closely, they fail to detect rare, dangerous attacks like DNS Spoofing because they focus on getting high accuracy on majority benign traffic."*

---

## Slide 4: Proposed Solution & Key Novelties

*   **Slide Header**: Our Novel Approach: RA-RL-IDS
*   **Key Novelties**:
    1.  **Cost-Sensitive Reward Shaping (Gap B Solution)**:
        *   Replaces standard flat $+1$/$-1$ rewards with **Inverse Class-Frequency Weights ($w_c$)**.
        *   Imposes scaled penalties on false negatives for rare attacks ($r = -w_{true\_class} \times \text{penalty}$), forcing the RL agent to prioritize minority threat detection.
    2.  **Hardware-Aware Post-Training Quantization (Gap A Solution)**:
        *   Applies **Dynamic INT8 Quantization** to PyTorch LSTM and Linear layers, mapping 32-bit floats to 8-bit integers.
        *   Conducts **CPU-only edge benchmarking** to verify inference latency, RAM footprint, and storage footprint under edge conditions.

### Speaker Notes (What to say out loud):
> *"To solve both gaps, our project introduces two core novelties. First, we reformulate the Reinforcement Learning reward function using inverse class-frequency weights, heavily penalizing the agent whenever it misses a rare attack. Second, we apply post-training INT8 dynamic quantization to compress the neural network by over 70%, ensuring it runs in real time on a commodity CPU."*

---

## Slide 5: System Architecture & Proposed Workflow

*   **Slide Header**: Technical Pipeline & Agent Architecture
*   **System Components**:
    *   **Data Pipeline & Preprocessing**:
        *   Standard Z-score scaling and median imputation.
        *   **Mutual Information Feature Selection (MIFS)** to extract the top 25 high-influence network flow features.
        *   Strict stratified splitting (Train/Val/Test) to prevent data leakage.
    *   **Gymnasium RL Environment (`IoMTIDSEnv`)**:
        *   Formulates flow classification as a sequential Markov Decision Process (MDP).
    *   **Hybrid 1D CNN-LSTM Deep Q-Network (DQN)**:
        *   **1D CNN**: Extracts spatial header correlations across packet fields.
        *   **LSTM**: Models temporal sequence dependencies across network flows.
        *   **DQN Head**: Evaluates Q-values across 16 traffic classes with target-network stabilization.

### Speaker Notes (What to say out loud):
> *"Our architecture consists of three modular components. The data pipeline cleans the network flows and uses Mutual Information to select the top 25 most informative features. The Gymnasium environment feeds these flows sequentially to our hybrid neural network, where a 1D CNN extracts spatial packet header relationships and an LSTM captures temporal sequence dependencies to make 16-class predictions."*

---

## Slide 6: Experimental Setup & Validation Strategy

*   **Slide Header**: Dataset, Metrics, and Benchmark Methodology
*   **Benchmark Dataset**:
    *   **CICIoMT2024**: Modern benchmark dataset containing 16 classes (1 Benign + 15 Attack types across Wi-Fi, BLE, and MQTT protocols).
*   **Evaluation Metrics**:
    *   *Classification Metrics*: Per-Class Recall (focusing on minority classes), Macro F1-score, Weighted F1-score, and Confusion Heatmaps.
    *   *Hardware Edge Metrics*: Binary Model Size (MB), Peak Memory Footprint (KB) via `tracemalloc`, and Average Per-Sample CPU Latency (ms).
*   **Real-Time Budget Target**:
    *   Latency requirement $< 50\text{ ms}$ per sample; Storage footprint target $< 500\text{ KB}$.

### Speaker Notes (What to say out loud):
> *"We validate our approach using the CICIoMT2024 multi-protocol healthcare dataset. We evaluate classification performance using Per-Class Recall to prove that rare attack detection improves, and we benchmark hardware efficiency on CPU to measure storage size, RAM usage, and per-packet execution latency against a real-time 50-millisecond budget."*

---

## Slide 7: Expected Outcomes, Roadmap & Conclusion

*   **Slide Header**: Projected Deliverables & Project Impact
*   **Expected Results**:
    *   **Significant Gain in Minority Recall**: Multi-fold improvement in detecting rare attacks (e.g., DNS Spoofing, OS Scanning, MQTT exploits).
    *   **Lightweight Deployable Model**: $>70\%$ reduction in binary model footprint (compressing from ~520 KB down to ~142 KB).
    *   **Real-Time Processing**: Sub-5 ms inference latency per packet on CPU hardware with negligible accuracy loss ($<0.1\%$).
*   **Project Roadmap**:
    *   *Phase 1*: Environment setup, data pipeline, and Gym env construction *(Completed)*.
    *   *Phase 2*: Model training, reward shaping tuning, and quantization benchmarking *(Completed)*.
    *   *Phase 3*: Deployment on physical edge hardware (Raspberry Pi gateway) & final documentation *(Upcoming)*.

### Speaker Notes (What to say out loud):
> *"In conclusion, RA-RL-IDS provides a practical, deployable solution for IoMT network security. By combining inverse-frequency reward shaping with INT8 quantization, we achieve a model that detects stealthy targeted attacks while being small enough (142 KB) and fast enough (under 4 ms) to run on low-cost edge gateways. Thank you, and I am now open to your questions and feedback."*
