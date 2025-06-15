import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Load your saved model
model = load_model("mobilenet_model.keras")

# Load the same validation data used in training
from load_data import val_data

# --------- 1. Plot Accuracy & Loss (Optional if you saved history) ---------
# You can skip plotting accuracy/loss here if you didn't save 'history'

# --------- 2. Generate Predictions & Confusion Matrix ---------
Y_pred = model.predict(val_data)
y_pred = np.argmax(Y_pred, axis=1)
y_true = val_data.classes
class_names = list(val_data.class_indices.keys())

# Print classification report
print("Classification Report:\n")
print(classification_report(y_true, y_pred, target_names=class_names))

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(12, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()


# from sklearn.metrics import classification_report, confusion_matrix
# import numpy as np
# from tensorflow.keras.models import load_model

# # Predict labels on validation set
# val_preds = model.predict(val_data)
# y_pred = np.argmax(val_preds, axis=1)
# y_true = val_data.classes

# # Print evaluation metrics
# print(classification_report(y_true, y_pred, target_names=val_data.class_indices.keys()))
