# credit-scoring-deep-learning
"An AI system using Deep Learning to predict loan default risk and improve credit decision-making."
#  AI-Powered Credit Scoring System

##  Overview
This project implements a **Deep Learning** model to predict credit risk. Using Artificial Neural Networks (ANN), the system evaluates whether a loan applicant is likely to default or pay back, helping financial institutions make data-driven decisions.
##  Technical Approach
### 1. Data Preprocessing & Scaling
* **StandardScaler:** Applied to normalize features like income and loan amount, ensuring the model isn't biased towards larger numerical values.
* **LTI Ratio:** Engineered the Loan-to-Income (LTI) feature as a primary indicator of financial stress.

### 2. Handling Data Imbalance
In credit datasets, defaults are rare. I focused on:
* Balancing the classes to improve the model's ability to detect risky clients.
* Optimizing for **Recall** to minimize "False Negatives" (missed defaults).
* ##  Model Architecture
The model is a Sequential Neural Network built with **Keras** and **TensorFlow**:
* **Input Layer:** Designed for 5 financial features (Income, Loan Amount, Years of Experience, LTI, and Age).
* **Hidden Layers:** Integrated with **ReLU** activation functions and **Dropout (0.2)** layers to prevent overfitting.
* **Output Layer:** Utilizes a **Sigmoid** activation function to predict the probability of default (range 0-1).

##  Future Improvements
* Integrating **SHAP** (Explainable AI) to provide reasons for specific loan rejections.
* Expanding the dataset to include macro-economic indicators for better accuracy.
##  How to Run
1. **Clone the repository:**
   ```bash
   git clone [https://github.com/your-username/credit-scoring-deep-learning.git](https://github.com/your-username/credit-scoring-deep-learning.git)
   pip install tensorflow pandas scikit-learn numpy
   python train.py
