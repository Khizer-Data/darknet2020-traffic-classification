# darknet2020-traffic-classification
Machine learning models applied to the Darknet 2020 network traffic dataset for multi-class classification, achieving up to 90% accuracy with Random Forest.

# 🚀 Darknet 2020 - Network Traffic Classification

![License](https://img.shields.io/badge/license-MIT-green) 
![Python](https://img.shields.io/badge/python-3.10-blue)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Completed-success)
![Status](https://img.shields.io/badge/status-Finished-brightgreen)

## 🌟 **Overview**
This project applies multiple machine learning algorithms to the **[Darknet 2020 dataset](https://www.unb.ca/cic/datasets/darknet2020.html)**, focusing on network traffic classification for cybersecurity applications.  
The aim is to detect patterns, classify traffic, and identify potential anomalies with the best-performing model achieving **90% accuracy**.

---

## 🎯 **Project Objectives**
- ✅ Preprocess and merge multiple subsets of the Darknet 2020 dataset.  
- ✅ Apply machine learning models for multi-class classification.  
- ✅ Compare performance metrics including accuracy, precision, recall, and F1-score.  
- ✅ Identify the most efficient model for cybersecurity-related network traffic classification.

---

## 🧪 **Machine Learning Models & Results**

| Model                  | Accuracy | Precision (avg) | Recall (avg) | F1-Score (avg) |
|------------------------|:---------:|:---------------:|:------------:|:--------------:|
| 🔹 **Logistic Regression**   | 57.65%   | 0.63            | 0.58         | 0.57           |
| 🌲 **Random Forest**         | ⭐ **90.05%** | 0.90            | 0.90         | 0.90           |
| ⚡ **XGBoost**               | 82.01%   | 0.82            | 0.82         | 0.82           |
| 🤖 **Keras Sequential (DNN)**| 72.32%   | 0.72            | 0.72         | 0.72           |
| 🎛 **Bagging Classifier**    | 89.37%   | 0.90            | 0.89         | 0.89           |
| 🎯 **AdaBoost**              | 68.85%   | 0.72            | 0.69         | 0.69           |
| 🌊 **Gradient Boosting**     | 76.21%   | 0.78            | 0.76         | 0.76           |

---

## 📊 **Key Insights**
- ✅ **Random Forest** emerged as the top performer (**90.05% accuracy**), showcasing tree-based models' effectiveness.  
- ⚡ **XGBoost** provided a solid balance of performance (**82.01% accuracy**) and computational efficiency.  
- 🔥 **Deep Learning (Keras Sequential)** performed reasonably well (**72.32% accuracy**), with potential for improvement through hyperparameter tuning.  
- ⚠️ **Logistic Regression and AdaBoost** underperformed, highlighting the need for more complex models for this dataset.

---

## 📝 **Conclusion**
Tree-based ensemble models like **Random Forest** and **XGBoost** outperform other models in classifying network traffic data. Deep learning models can further improve performance with additional tuning, though simpler algorithms like Logistic Regression aren't well-suited for this complexity.

---

## 💡 **Technologies Used**
- 📚 **Python 3.10**  
- 🔍 **scikit-learn**  
- ⚡ **XGBoost**  
- 🤖 **TensorFlow & Keras**  
- 🌲 **Random Forest, Gradient Boosting**  
- 🛠 **Pandas, NumPy, Matplotlib, Seaborn**

---

## 🖋 **Author**  
👨‍💻 *[Your Name Here]*  

📬 *Feel free to contribute or open issues for discussions!*  

---

⭐ **If you like this project, give it a star!** ⭐

