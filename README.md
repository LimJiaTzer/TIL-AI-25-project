# 🏆 BRAINHACK TIL-AI 2025 Challenges Showcase 🚀

Welcome to our team's repository for the **TIL-AI 2025 Challenges**!  
This project showcases our collective effort in tackling a diverse set of AI and machine learning problems—from audio processing and computer vision to reinforcement learning and optimization.

We are proud to present our **innovative solutions** and the **cutting-edge models** we implemented.

📄 **Challenge Requirements:**  
You can find the detailed specifications for each challenge [here on the official wiki](https://github.com/til-ai/til-25/wiki/Challenge-specifications).  

![photo_2025-06-12_23-08-46](https://github.com/user-attachments/assets/b2180a0f-a471-473f-b5d0-4691637ec267)  

---

## 🏅 Competition Achievement

🎉 We are proud to have placed in the **Top 8** teams at the **TIL-AI 2025** competition!

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

Below is a breakdown of the challenges we tackled and the methodologies we employed to solve them.

---

### 1. 🎤 Automatic Speech Recognition (ASR)

**Challenge:** Accurately transcribe spoken audio into text.

**Our Approach:**  
We used a dual-model architecture:
- **Primary Model:** [`nvidia/parakeet-tdt-0.6b-v2`](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2) – high accuracy and low latency.
- **Secondary Model:** [`distil-whisper/distil-large-v3.5`](https://huggingface.co/distil-whisper/distil-large-v3.5) – robust performance across diverse accents and noise levels.

---

### 2. 👁️ Computer Vision (CV)

**Challenge:** Detect and identify specific objects within images with high speed and accuracy.

**Our Approach:**  
- **Model:** [`YOLOv8`](https://docs.ultralytics.com/models/yolov8/) – fine-tuned on our custom dataset.
- **Optimization:**  
  Converted the PyTorch model to a **TensorRT `.engine`** file to:
  - Minimize latency.
  - Maximize throughput on GPU during inference.

---

### 3. 📄 Optical Character Recognition (OCR)

**Challenge:** Extract text from scanned documents and images.

**Our Approach:**  
We implemented [`stepfun-ai/GOT-OCR-2.0-hf`](https://huggingface.co/stepfun-ai/GOT-OCR-2.0-hf), a transformer-based OCR model that excels in:
- Complex layouts.
- Challenging visual conditions.
- Multilingual and stylized text extraction.

---

### 4. 🤖 Reinforcement Learning: A Hybrid Agent System

**Challenge:** Develop an agent to navigate a complex environment guarded by intelligent opponents.

#### Part 1: Heuristic Guard Agent

- **Architecture:** Finite State Machine (FSM).
- **States:**
  - `PATROLLING`: Follows fixed waypoints.
  - `HUNTING`: Pursues detected Scouts.
  - `ROAMING`: Searches areas based on last known Scout location.
- **Pathfinding:** Efficient navigation via Dijkstra's algorithm on a precomputed graph.

#### Part 2: Hybrid Reinforcement Learning Scout

- **Model:** Deep Q-Network (DQN) with a CNN-based observation pipeline.
- **Hybrid Architecture:**
  - **Evasion Layer:** Overrides DQN if a Guard is nearby.
  - **Anti-Repetition Layer:** Prevents action loops using movement memory.
  - **Policy Layer:** Default DQN actions when the agent is safe.

This blend of heuristics and deep learning results in a resilient and adaptive agent.

---

### 5. 🧩 Surprise Challenge: Document Reassembly

**Challenge:** Reconstruct a shredded document from vertical strips.

**Our Approach:**  
- Modeled the task as a **Traveling Salesperson Problem (TSP)**.
- **Scoring:** Used pixel-based dissimilarity (e.g., Sum of Squared Differences) between strip edges.
- **Solution:**
  - Generated a cost matrix.
  - Fed it into a TSP solver (e.g., LKH).
  - Output the optimal order of strips for accurate reconstruction.

---

## 🙌 Thank You!

Thank you for visiting our repository!  
We hope our work offers inspiration and valuable insights into solving complex AI challenges.

Feel free to ⭐ star, 🍴 fork, or reach out if you're interested in learning more or collaborating!

