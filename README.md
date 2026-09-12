# ⚡ Delhi SLDC - 3D AI Grid Command Center

A predictive power grid analytics and feeder balancing dashboard designed for the **Delhi State Load Despatch Centre (SLDC)** using Streamlit, Plotly, Scikit-Learn, and XGBoost.

## 🌟 Key Features
- **3D Load Topology Surface:** Interactive 3D mesh mapping grid demand (MW) against hour of day and ambient temperature.
- **Multi-Model AI Engine:** Trains Random Forest, Gradient Boosting, XGBoost, and Ridge Regression models on demand patterns.
- **Rooftop Solar Offset:** Real-time generation deduction based on solar capacity and hourly solar irradiance curves.
- **Heatwave Stress Testing:** Climate offset tool simulating ambient temperature surges (+1°C to +6°C).
- **Discom Feeder Load Shedding:** Automated emergency shedding priority matrix across BRPL, TPDDL, BYPL, and NDMC.

## 📂 Project Structure
```text
delhi-sldc-dashboard/
├── .streamlit/
│   └── config.toml
├── app.py
├── model_engine.py
├── requirements.txt
├── .gitignore
└── README.md
🚀 Quick Start
Bash
# Clone repository
git clone [https://github.com/YOUR_USERNAME/delhi-sldc-dashboard.git](https://github.com/YOUR_USERNAME/delhi-sldc-dashboard.git)
cd delhi-sldc-dashboard

# Install dependencies
pip install -r requirements.txt

# Launch app
streamlit run app.py
