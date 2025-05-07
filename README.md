## 🔍 Deepfake Detection Using Graph Convolutional Networks (GCNs)

In this project, we tackle the growing threat of deepfakes—AI-generated fake videos that can spread misinformation and violate personal privacy—by moving beyond traditional CNN-based detection methods. Instead, we leverage **Graph Convolutional Networks (GCNs)** to capture the **spatial and geometric relationships between facial landmarks**, providing a more nuanced understanding of facial structures.

### 🚀 What We Did:

* ✅ **Built a deepfake detection pipeline** that transforms video frames into facial landmark graphs.
* 🧠 **Applied GCNs** to model facial dependencies, outperforming CNNs in detecting subtle manipulations.
* 📊 **Achieved over 90% accuracy**, with high precision (92%), recall (89%), and F1-score (90.5%) on the Deepfake Detection Challenge dataset.
* 👁️‍🗨️ **Visualized facial graphs**, highlighting key regions (eyes, mouth) where manipulation often occurs.

### 🛠️ Skills & Technologies:

* Graph Convolutional Networks (GCNs)
* Facial landmark detection
* Deep learning and computer vision
* Video preprocessing & frame extraction
* Model evaluation (Accuracy, Precision, Recall, F1-Score)
* Python, PyTorch (or TensorFlow, depending on your stack)

This work demonstrates the potential of **graph-based deep learning** in media forensics and opens the door to more robust, interpretable deepfake detection systems.


### 📁 Dataset: Deepfake Detection Challenge (DFDC)

The **DFDC dataset** is a large-scale benchmark developed by Facebook AI in collaboration with industry partners to advance deepfake detection research. It comprises over **100,000 video clips** featuring both real and AI-manipulated content, created using various deepfake generation techniques. The dataset includes:

* **Diverse Subjects**: Videos of **3,426 paid actors**, ensuring a wide range of facial features, expressions, and backgrounds.
* **Varied Manipulations**: Deepfakes generated using multiple face-swapping methods, including GAN-based and non-learned techniques, to simulate real-world scenarios.
* **Balanced Dataset**: A mix of authentic and manipulated videos to train models effectively.
* **Ethical Considerations**: All participants provided consent for their likenesses to be used and altered in the dataset.

This dataset serves as a comprehensive resource for training and evaluating models aimed at detecting deepfake videos. For more details and access to the dataset, visit the [DFDC Kaggle page](https://www.kaggle.com/competitions/deepfake-detection-challenge).



### 🧪 Methodology Overview

Our deepfake detection approach consists of the following key steps:

1. **Frame Extraction**: Extract individual frames from video samples.
2. **Facial Landmark Detection**: Use a landmark detector to identify key facial points (e.g., eyes, nose, mouth).
3. **Graph Construction**: Represent each face as a graph, where nodes are landmarks and edges denote spatial relationships.
4. **GCN Processing**: Pass the facial graphs through a multi-layer Graph Convolutional Network to learn spatial and geometric features.
5. **Classification**: Use the extracted features to classify each frame as **real** or **fake**.

This graph-based method enables the model to understand complex facial structures and detect subtle manipulations typical in deepfake videos.



