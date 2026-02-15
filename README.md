🚀 GAN-Assisted Hybrid Deep Learning Ensemble for Multi-Class Network Intrusion Detection
📌 Overview
This project presents a GAN-enhanced Hybrid Deep Learning Ensemble Intrusion Detection System (IDS) designed to improve multi-class intrusion detection performance under severe class imbalance conditions.

The system integrates:

Generative Adversarial Networks (GAN) for minority attack synthesis

Deep Neural Network (DNN) with Focal Loss

Autoencoder-based feature compression

Cost-sensitive ensemble learning

Stacking meta-learning

Confidence-based rare-class correction

Hierarchical IDS architecture

The proposed system is evaluated on UNSW-NB15 and NSL-KDD datasets.

🎯 Key Features
✅ GAN-based adaptive class balancing

✅ Hybrid Deep Feature Extraction (DNN + Autoencoder)

✅ Cost-sensitive Random Forest & Logistic Regression

✅ Stacking Meta-Learner (HistGradientBoosting)

✅ Hierarchical Binary + Multi-class IDS

✅ Confidence-based rare class correction

✅ Explainable AI (Feature Importance Analysis)

🧠 System Architecture
Raw Dataset
    ↓
Preprocessing & Leakage Removal
    ↓
GAN-Based Balancing
    ↓
DNN + Autoencoder Feature Extraction
    ↓
Hybrid Deep Feature Concatenation
    ↓
Cost-Sensitive ML Models
    ↓
Stacking Meta-Learning
    ↓
Hierarchical + Confidence-Based Prediction
📊 Datasets Used
UNSW-NB15 (Multi-Class Intrusion Detection)

NSL-KDD (Binary Intrusion Detection)

Both datasets were preprocessed with:

Label Encoding

Standard Scaling

Stratified Train-Test Split (70:30)

📈 Results
🔹 Binary Classification (NSL-KDD)
Accuracy: ~98%

High Attack Recall

Improved F1 Score

🔹 Multi-Class Classification (UNSW-NB15 – 10 Classes)
Accuracy: 81.77%

Macro F1: ~0.47+

Weighted F1: ~0.79+

Performance improved through:

GAN-based oversampling

Focal Loss optimization

Hybrid feature stacking

🏗 Project Structure
GAN-Hybrid-NIDS/
│
├── src/                         # All source code files
│   ├── preprocess.py
│   ├── train_gan.py
│   ├── generate_balanced.py
│   ├── train_ensemble.py
│   ├── m2_h1_train_dnn.py
│   ├── m2_h2_train_autoencoder.py
│   ├── m2_h5_stacking_meta.py
│   └── ...
│
├── .gitignore
├── requirements.txt
└── README.md
⚙️ Installation
Clone the repository:

git clone https://github.com/Mann65121/GAN-Hybrid-NIDS.git
cd GAN-Hybrid-NIDS
Create virtual environment:

python3 -m venv venv
source venv/bin/activate
Install dependencies:

pip install -r requirements.txt
▶️ How to Run
Step 1 – Preprocessing
python src/preprocess.py UNSW_NB15
Step 2 – Train GAN
python src/train_gan.py UNSW_NB15
Step 3 – Generate Balanced Data
python src/generate_balanced.py UNSW_NB15
Step 4 – Train Deep Models
python src/m2_h1_train_dnn.py
python src/m2_h2_train_autoencoder.py
Step 5 – Train Ensemble
python src/m2_h5_stacking_meta.py
Step 6 – Evaluate
python src/m2_h6_evaluate.py
🧪 Evaluation Metrics
Accuracy

Precision

Recall

F1 Score

Macro F1

Weighted F1

Macro F1 ensures rare class evaluation, while Weighted F1 reflects real-world distribution.

🔍 Explainability
Random Forest feature importance analysis was performed to identify top contributing features for intrusion detection.

🏆 Achievements
Hybrid GAN-based IDS successfully implemented

Multi-class intrusion detection achieved (10 classes)

Research paper drafting in progress

Prototype ready for real-time deployment extension

📚 References
Tavallaee et al., NSL-KDD Dataset Analysis (2009)

Moustafa & Slay, UNSW-NB15 Dataset (2015)

Goodfellow et al., Generative Adversarial Networks (2014)

Recent GAN-based IDS Research Literature

👨‍💻 Authors
Manav Bhatt
Prajjwal Sharma

Department of Computer Engineering
PBL Project – Hybrid AI IPS Integration
