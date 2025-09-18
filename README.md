# 🚗₹✨ Used Car Price Prediction App

[![Streamlit](https://img.shields.io/badge/streamlit-powered-green)](https://streamlit.io/)  
[![Python](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)  
[![License](https://img.shields.io/badge/license-MIT-lightgrey)](LICENSE)

---

## 📖 Project Description

This Streamlit application predicts the **selling price of used cars** based on a trusted dataset and machine learning model. It leverages powerful **Gradient Boosting Regression** to provide highly accurate price estimates in the Indian used car market context, assisting buyers and sellers to make informed decisions with confidence.

The model considers features such as:

- Car manufacturing year (age)  
- Kilometers driven  
- Fuel type (Petrol, Diesel, CNG, LPG, Electric)  
- Seller type (Individual, Dealer, Trustmark Dealer)  
- Transmission type (Manual, Automatic)  
- Ownership history  

---

## ⚙️ Features

- 🚀 Fast and interactive web interface using Streamlit  
- 🎯 Accurate price predictions using Gradient Boosting Regressor  
- 🔍 Real-time input for customizable car details  
- 🛠️ Scalable pipeline with standardized feature preprocessing and encoding  
- 📊 Clean, easy-to-understand UI with meaningful outputs  

---
## 🚀 Demo

The Streamlit app is [Used Car Prediction](https://ml-used-car-price-prediction.streamlit.app/)

**Screen Capture**

<img width="456" height="362" alt="image" src="https://github.com/user-attachments/assets/08191dd6-a0ce-4ce6-b5b8-7605fb04a922" />
<img width="456" height="362" alt="image" src="https://github.com/user-attachments/assets/0940f546-a157-4738-bdbb-cfe50f221d1c" />
<img width="456" height="362" alt="image" src="https://github.com/user-attachments/assets/49b4983f-85be-4992-bef5-468dc01b24be" />


---

## 📁 File Structure


├── app.py # Streamlit app script

├── gb_model.pkl # Trained Gradient Boosting model

├── scaler.pkl # Feature scaler

├── requirements.txt # Python dependencies

├── README.md # Project description and instructions

└── data/ # (Optional) Dataset or other supporting files


---

## 💡 How to Use

1. Open the app in your browser via Streamlit server.  
2. Input your car details on the sidebar and main UI fields.  
3. Click the **Predict Selling Price** button.  
4. View the predicted resale price instantly.

---

## 📊 Model Performance

| Model               | R2 Score | MAE  | RMSE  |
|---------------------|----------|------|-------|
| Gradient Boosting    | 0.5388   | 0.43 | 0.71  |

*Gradient Boosting outperforms other tested models, delivering more reliable price predictions.*

---

## 🤝 Contributing

Contributions are welcome! Feel free to open issues or submit pull requests to improve the app or extend functionality.

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgements

- Dataset sourced from CarDekho  
- Inspired by machine learning techniques in used car price prediction research

---

## 📬 Contact

For questions or collaboration, please contact:

👤 **Developer Name**  
📧 sriram.tsbuss@gmail.com  
🔗 [LinkedIn Profile](https://www.linkedin.com/in/sriram-ts-73030614/)  

---

Made with ❤️ using Streamlit and Python  


