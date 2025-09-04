# 📈 Stock Market Prediction Using Hybrid Deep Learning

This project predicts stock prices by combining **time-series data** with **news sentiment analysis** using a hybrid deep learning approach:

- 🌀 **Transformer** for historical stock trends  
- 📰 **GPT-2 embeddings** for extracting sentiment from news  
- ⚡ **Quantum-Inspired Neural Network (QINN)** for modeling non-linear market patterns  

The final model leverages **historical data + news headlines** to improve stock price prediction accuracy.

---

## 🚀 Features

- 📊 Time-Series Transformer for trend learning  
- 📰 GPT-2 embeddings for financial news analysis  
- ⚡ QINN for quantum-inspired modeling  
- 🎯 Predicts future stock price movements  
- 🌐 Flask web app for uploading CSV files & interactive results  
- 📉 Visualization of **true vs predicted stock prices**  

---

## 🛠️ Technologies Used

- **Python** 🐍  
- **PyTorch** ⚡  
- **HuggingFace Transformers (GPT-2)**  
- **Scikit-learn & Pandas**  
- **Flask** (Web App)  
- **Matplotlib** (Graph Visualization)  

---

## 📊 How It Works

1. Upload **historical stock data** (CSV) + **current stock data** via the Flask app.  
2. News headlines are processed using **GPT-2 embeddings** (or fallback features).  
3. Stock data is normalized and fed into the **Time-Series Transformer**.  
4. A hybrid model (**Transformer + QINN**) predicts future stock prices.  
5. Results are shown with **interactive line charts** in the web UI.  

---

## 📁 Dataset

- **Historical stock data** (CSV)  
- **News dataset** (financial headlines, e.g., from Kaggle / Yahoo Finance)  

