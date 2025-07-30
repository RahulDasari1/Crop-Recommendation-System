# 🌾 Crop Recommendation System

A machine learning based web application that recommends optimal crops based on soil nutrients and environmental conditions.

<img width="1889" height="902" alt="image" src="https://github.com/user-attachments/assets/85276e62-3ec9-4b7a-b5d2-980e95cb7d5c" />


## ✅ Features

* Predict crop suitability using environmental and soil data:

  * Nitrogen (N), Phosphorus (P), Potassium (K)
  * Soil pH, temperature, humidity, and rainfall
* Built with a trained classification model for automated crop suggestion
* Web app interface via `Flask` (`app.py`)
* Pretrained model and scalers included (`model.pkl`, `minmaxscaler.pkl`, `standscaler.pkl`)
* Example Jupyter notebook for model exploration: `Crop Recommendation Using Machine Learning.ipynb`
  ([GitHub][1], [GitHub][2], [GitHub][3])

## 📂 Repository Structure

```
Crop_Recommendation/
├── app.py                     # Flask application for predictions
├── model.pkl                  # Trained ML model
├── minmaxscaler.pkl           # Preprocessing scaler (MinMax)
├── standscaler.pkl            # Preprocessing scaler (Standard)
├── Crop Recommendation Using Machine Learning.ipynb  # Notebook with EDA & model training
├── Crop_recommendation.csv    # Dataset used to train the model
├── templates/                 # HTML templates for UI (e.g. index.html)
└── static/                    # Static assets (CSS, JS, images)
```

## 🚀 Getting Started

### Prerequisites

* Python 3.7+
* Virtual environment recommended

### Install Dependencies

```bash
pip install -r requirements.txt
```

*(If a `requirements.txt` isn’t present, install `Flask`, `pandas`, `scikit-learn`, `numpy`, etc.)*

### Run the Web App

```bash
python app.py
```

Then open your browser at `http://127.0.0.1:5000/` to enter soil and weather details and receive crop suggestions.

## 🧪 How It Works

1. User inputs soil nutrients and climate parameters via the web UI.
2. `app.py` loads `model.pkl` and appropriate scalers.
3. Input data is scaled then fed into the model.
4. Model outputs a recommended crop based on trained patterns.
5. The app displays the result immediately in the browser.

## 📝 Notebook & Model

* The notebook contains exploratory data analysis, model training steps, and performance metrics.
* It demonstrates how the dataset was preprocessed, normalized (MinMax or Standard), and how the classifier was trained.
* The final model and scalers used in production are included in the repo.
  ([GitHub][1])

## 📈 Data

The dataset `Crop_recommendation.csv` includes features such as soil nutrients (N, P, K), temperature, humidity, soil pH, and rainfall for each observation.

## 🛠️ Technologies Used

* **Python** for backend and machine learning
* **Flask** for building the prediction web interface
* **scikit-learn** for training and using classification models
* **pandas**, **numpy** for data manipulation
* **Jupyter Notebook** for EDA and model development
  ([GitHub][1])

## 🎯 Usage Example

1. Navigate to the home page and fill in the required inputs:

   * N, P, K, temperature, humidity, pH, rainfall
2. Submit the form.
3. View the predicted optimal crop for given inputs.

## 👥 Contributing

Contributions are welcome! Please:

1. Fork the repo
2. Create a feature branch (e.g. `feature/new-model`)
3. Submit a pull request

Areas for potential improvement:

* Support for weather API integration
* Enhancing UI design and responsiveness
* Multi-model comparison (e.g. decision tree, random forest, XGBoost)
* Support for deploying on cloud platforms (Heroku, AWS, etc.)

## 📜 License

Distributed under the MIT License. See `LICENSE` file for more details (if applicable).

## 💬 Questions?

Feel free to open an issue or contact me through GitHub. I’d love to hear feedback or suggestions!
