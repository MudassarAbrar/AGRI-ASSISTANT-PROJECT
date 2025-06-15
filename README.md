# 🌿 AI-Powered Agricultural Assistant

A minimal viable product (MVP) to detect plant diseases from leaf images, provide weather-based advice, and offer a simple farmer-friendly interface using Streamlit.

---

## 📁 Folder Structure

```plaintext
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> 223a68e8 (Initial Commit)
AGRI-ASSISTANT/
├── data/
│   └── PlantVillage/                 # (Optional) Sample dataset of plant leaves
│
├── disease_model/                   # Model training, evaluation, and prediction code
│   ├── AGRI-ASSISTANT.code-workspace  # VSCode workspace config (optional)
│   ├── class_indices.json           # Mapping of class labels to indices
│   ├── class_labels.pkl             # Serialized label encoder or class info
│   ├── evaluate.py                  # Script to evaluate model performance
│   ├── load_data.py                 # Data loading and preprocessing functions
│   ├── main.py                      # Optional entry point for training
│   ├── model.py                     # Model architecture definition
│   ├── plant_disease_model.keras    # Saved trained Keras model
│   ├── predict_image.py             # Script to make single image predictions
│   ├── train_mobilenet.py           # Script to train MobileNet model
│   └── __pycache__/                 # Python bytecode cache (ignored)
│
├── streamlit_app/
│   └── app.py                       # Streamlit frontend for disease prediction
│
├── .env                             # Environment variables (ignored)
├── .gitignore                       # Git ignore rules
├── .python-version                  # Python version used (e.g., 3.10)
├── README.md                        # Project documentation and usage guide
├── requirements.txt                 # Python dependencies
└── .venv/                           # Python virtual environment (ignored)
