# 🏆 BRAINHACK TIL-AI 2025 Challenges Showcase 🚀

Welcome to our team's repository for the **TIL-AI 2025 Challenges**!  
This project showcases our collective effort in tackling a diverse set of AI and machine learning problems—from audio processing and computer vision to reinforcement learning and optimization.

We are proud to present our **innovative solutions** and the **cutting-edge models** we implemented.

---

## 👥 Our Team

This project was a collaborative effort by a dedicated and passionate team:

- Spencer  
- **Jia Tzer**  
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
- **Primary Model:** [NVIDIA Parakeet](https://github.com/NVIDIA/NeMo) – known for high accuracy and efficiency.
- **Secondary/Benchmark Model:** [OpenAI Whisper](https://github.com/openai/whisper) – robust across diverse audio conditions.

This setup allowed us to ensure high-quality transcriptions across a variety of inputs and noise levels.

---

### 2. 👁️ Computer Vision (CV)

**Challenge:** Detect and identify specific objects within images with high speed and accuracy.

**Our Approach:**  
- **Model:** YOLOv8 – fine-tuned on a custom dataset.
- **Optimization:**  
  We converted the trained PyTorch model to a **TensorRT** `.engine` file to:
  - Reduce inference latency.
  - Accelerate evaluation on compatible GPU hardware.

---

### 3. 📄 Optical Character Recognition (OCR)

**Challenge:** Extract text from scanned documents and images.

**Our Approach:**  
We deployed the **stepfun-ai/GOT-OCR-2.0-hf** transformer model, which excels at:
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
