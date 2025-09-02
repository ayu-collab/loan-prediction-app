#  Loan Approval Prediction App

This project is a **Machine Learning-powered Loan Approval Prediction App** built with **Streamlit**.  
It predicts whether a loan application will be **Approved** or **Rejected** based on applicant details.

---

##  Project Overview
- **Data**: Loan dataset (historical loan applications).  
- **Model**: Linear Regression.  
- **Features**: Includes applicant income, loan amount, dependents, property area, etc.  
- **Frontend**: Streamlit app for interactive predictions.  

---

## Running the App Locally

1. **Clone the repository**  
   ```bash
   git clone <your-repo-url>
   cd Project
   ```

2. **Install dependencies**  
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit app**  
   ```bash
   streamlit run app.py
   ```

4. **Open browser**  
   Streamlit will provide a local URL (e.g., `http://localhost:8501`). Open it to use the app.

---

## Repository Structure

```
Project/
│── data.csv                 # Dataset
│── script.ipynb             # EDA + Model training notebook
│── model_rf.pkl             # Saved model
│── feature_list.pkl         # Feature order used in training
│── preproc_meta.pkl         # Preprocessing metadata
│── _encoders.pkl            # Encoders used for categorical variables
│── scaler.pkl               # Scaler used for feature normalization

│── app.py                   # Streamlit application
│── requirements.txt         # Dependencies
│── README.md                # Project documentation

```

---

##  Output
<img width="1645" height="823" alt="Screenshot 2025-09-02 114042" src="https://github.com/user-attachments/assets/7794250e-20cd-4af6-91e7-b307c1c21c03" />
<img width="1600" height="708" alt="Screenshot 2025-09-02 114117" src="https://github.com/user-attachments/assets/7a1a901c-217f-4e10-9af0-3ede2439a184" />

---

##  Acknowledgements
- Dataset: https://www.kaggle.com/datasets/ninzaami/loan-predication 
- Tools: Python, Scikit-learn, Streamlit, Pandas, Numpy  

