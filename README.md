# Financial Decision Support System (DUO Planner) 📊

A data-driven web application aimed at optimizing student debt repayment strategies versus investing scenarios (S&P 500 / All-World ETF).

## 🎯 Project Goal
To determine the optimal financial strategy for Dutch graduates by modeling the trade-off between paying off student debt (DUO) and investing in the market. The model accounts for complex variables such as Dutch tax brackets (Box 3), purchasing power parity (inflation), and compounding interest over a 50-year horizon.

## 🛠 Tech Stack
* **Language:** Python 🐍
* **Visualization:** Plotly & Dash (Interactive Dashboarding)
* **Math:** NumPy (Vectorized calculations for performance)
* **Simulation:** Monte Carlo Simulations (500+ runs)

## 🚀 Key Features
* **Monte Carlo Simulation:** Simulates 500 potential market futures to assess risk profiles (P10, P50, P90).
* **Interactive Dashboard:** Users can adjust inflation, interest rates, and income growth to see real-time impact on Net Worth.
* **Object-Oriented Design:** Modular `Portfolio` classes handle the complex logic for loan amortization and tax calculations.
* **Vectorized Operations:** Uses NumPy arrays instead of loops for fast scenario generation.

## 📦 How to Run
1. Clone the repository
2. Install dependencies:
   pip install -r requirements.txt
3. Run the application:
  python app.py
4. Open your browser at `http://127.0.0.1:8051/`
