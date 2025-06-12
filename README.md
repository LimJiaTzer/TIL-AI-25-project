# 🏆 BRAINHACK TIL-AI 2025 Challenges Showcase 🚀

Welcome to our team's repository for the **TIL-AI 2025 Challenges**!  
This project showcases our collective effort in tackling a diverse set of AI and machine learning problems—from audio processing and computer vision to reinforcement learning and optimization.

We are proud to present our **innovative solutions** and the **cutting-edge models** we implemented.

📄 **Challenge Requirements:**  
You can find the detailed specifications for each challenge [here on the official wiki](https://github.com/til-ai/til-25/wiki/Challenge-specifications).

<img src="https://github.com/user-attachments/assets/254ef6cb-afb7-47ee-8348-4450a2fbc0a3" alt="Team Photo" width="800"/>
---

## 🏆 Competition Achievement

We are thrilled to announce that our team achieved a **Top 8** placement in the TIL-AI 2025 competition!

---

## 👥 Our Team

This project was a collaborative effort by a dedicated and passionate team:

- Spencer  
- Jia Tzer  
- Jia Jun  
- Gabriel  
- Joven  

---

## 🛠️ Projects & Solutions

Here's a breakdown of the challenges we tackled and the methodologies we employed to solve them.

---

### 1. 🎤 Automatic Speech Recognition (ASR)

**Challenge:** Accurately transcribe spoken audio into text.

**Our Approach:**  
We used a dual-model approach:
- **Primary Model:** [`nvidia/parakeet-tdt-0.6b-v2`](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2) – a high-accuracy, efficient model for transcription tasks.
- **Secondary/Benchmark Model:** [`distil-whisper/distil-large-v3.5`](https://huggingface.co/distil-whisper/distil-large-v3.5) – a lightweight, distilled version of Whisper offering robustness across diverse audio inputs.

---

### 2. 👁️ Computer Vision (CV)

**Challenge:** Detect and identify specific objects within images with high speed and accuracy.

**Our Approach:**  
- **Model:** [`YOLOv8`](https://docs.ultralytics.com/models/yolov8/) – fine-tuned on a custom dataset.
- **Optimization:**  
  We converted the trained PyTorch model to a **TensorRT** `.engine` file to:
  - Reduce inference latency.
  - Accelerate evaluation on compatible GPU hardware.

---

### 3. 📄 Optical Character Recognition (OCR)

**Challenge:** Extract text from scanned documents and images.

**Our Approach:**  
We deployed [`stepfun-ai/GOT-OCR-2.0-hf`](https://huggingface.co/stepfun-ai/GOT-OCR-2.0-hf), a transformer-based OCR model, which excels at:
- Recognizing complex text layouts.
- Extracting accurate text even under challenging visual conditions.

---

### 4. 🤖 Reinforcement Learning: A Hybrid Agent System

**Challenge:** Develop an agent to navigate a complex environment guarded by intelligent opponents.

#### Part 1: Heuristic Guard Agent (Non-Learning)

- **Design:** A Finite State Machine (FSM) governs Guard behavior.
- **States:**
  - `PATROLLING` — follows preset paths.
  - `HUNTING` — chases visible Scouts.
  - `ROAMING` — searches random areas.
- **Pathfinding:** Uses Dijkstra's algorithm on a precomputed map graph.

#### Part 2: Hybrid Reinforcement Learning Scout

- **Core Model:** Deep Q-Network (DQN) with a CNN visual input pipeline.
- **Hybrid Control Architecture:**
  - **Evasion Layer:** Rule-based override triggers when a Guard is nearby.
  - **Anti-Repetition Layer:** Heuristic avoids action loops by encouraging exploration.
  - **Default Mode:** DQN chooses the next action in safe conditions.

This hybrid design combines the pattern recognition of deep learning with the reliability of hand-coded heuristics, yielding a robust and adaptive agent.

---

### 5. 🧩 Surprise Challenge: Document Reassembly

**Challenge:** Reconstruct a shredded document from vertical strips.

**Our Approach:**  
- Treated as a **Traveling Salesperson Problem (TSP)**.
- **Cost Metric:** Dissimilarity score (e.g., Sum of Squared Differences) between strip edges.
- **Solution Strategy:**  
  - Constructed a cost matrix from edge comparisons.
  - Fed into a TSP solver (e.g., LKH or other libraries).
  - Output: Optimized strip order for accurate document reconstruction.

---

## 🙌 Thank You!

Thank you for visiting our repository!  
We hope our work offers inspiration and valuable insights into solving complex AI challenges.

Feel free to explore, fork, and reach out if you're interested in collaborating!

