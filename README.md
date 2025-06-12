# 🏆 BRAINHACK TIL-AI 2025 Challenges Showcase 🚀

Welcome to our team's repository for the TIL-AI 2025 Challenges! This project showcases our collective effort in tackling a diverse set of AI and machine learning problems, from audio processing and computer vision to reinforcement learning. We are proud to present our innovative solutions and the cutting-edge models we implemented.
![image](https://github.com/user-attachments/assets/510a730a-f46f-4c96-99c0-524f9d306454)

---

## 👥 Our Team

This project was a collaborative effort by a dedicated and passionate team.

* **Spencer**
* **Jia Tzer**
* **Jia Jun**
* **Gabriel**
* **Joven**

---

## 🛠️ Projects & Solutions

Here's a breakdown of the challenges we tackled and the methodologies we employed to solve them.

### 1. 🎤 Automatic Speech Recognition (ASR)
* **Challenge**: To accurately transcribe spoken audio into text.
* **Our Approach**: We leveraged a powerful combination of state-of-the-art ASR models. Our primary solution utilized the **NVIDIA Parakeet** model for its high accuracy and efficiency, with the **OpenAI Whisper** model serving as a robust secondary or comparative model. This dual-model approach allowed us to ensure high-quality transcriptions across various audio inputs.

### 2. 👁️ Computer Vision (CV)
* **Challenge**: To detect and identify specific objects within images with high speed and accuracy.
* **Our Approach**: We finetuned a **YOLOv8** model, a state-of-the-art object detection algorithm. After achieving high accuracy through training on a custom dataset, we focused on inference optimization.
* **Performance Enhancement**: To dramatically speed up model prediction, the trained PyTorch model was converted into a **TensorRT engine file (`.engine`)**. This step optimizes the model for the specific GPU architecture, leading to a substantial reduction in latency and a significant performance boost during evaluation.

### 3. 📄 Optical Character Recognition (OCR)
* **Challenge**: To extract text from scanned documents and images.
* **Our Approach**: For this task, we implemented the **`stepfun-ai/GOT-OCR-2.0-hf`** model. This powerful, transformer-based model excels at recognizing text in complex layouts and challenging visual conditions, providing highly accurate text extraction.

### 4. 🤖 Reinforcement Learning (RL)
* **Challenge**: To train an agent to make optimal decisions in a complex environment.
* **Our Approach**: We designed and implemented a **Heuristic Guard** system. This approach uses a set of smart, predefined rules to guide the RL agent, preventing it from making obviously poor decisions and significantly speeding up the learning process. It acts as a safety net, ensuring the agent explores the environment efficiently and effectively.

### 5. 🗺️ Scout: Advanced Reinforcement Learning
* **Challenge**: To navigate and succeed in a more advanced RL environment requiring sophisticated strategy.
* **Our Approach**: We developed a hybrid solution that combines the best of both worlds. Our "Scout" agent uses a **heuristic-guided, model-based Deep Q-Network (DQN)** with a **Convolutional Neural Network (CNN)** backbone. The CNN processes visual input from the environment, while the DQN learns the optimal action policy. The heuristic layer provides strategic oversight, leading to a highly intelligent and performant agent.

### 6. 🧩 Surprise: Document Reassembly
* **Challenge**: To reconstruct a shredded document from its vertical strips.
* **Our Approach**: Our strategy treats this as a variation of the Traveling Salesperson Problem (TSP). We first calculate a "dissimilarity" score between the right edge of one strip and the left edge of another using pixel differences (e.g., Sum of Squared Differences). These scores form a cost matrix, which we then feed into a **TSP solver (like LKH or a library-based one)** to find the optimal ordering of strips that minimizes the total dissimilarity, effectively reassembling the document.

---

Thank you for visiting our repository. We hope our work inspires and provides insight into solving complex AI challenges!
