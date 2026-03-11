import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay

# =========================
# Dataset distribution
# =========================
places = ['00', '02']
normal = [54, 8]
abnormal = [35, 8]

plt.figure(figsize=(6,4))
plt.bar(places, normal, label='Normal')
plt.bar(places, abnormal, bottom=normal, label='Abnormal')

plt.xlabel("Place")
plt.ylabel("Number of Events")
plt.title("Dataset Distribution")
plt.legend()

plt.savefig("dataset_distribution.png", dpi=300, bbox_inches='tight')
plt.close()


# =========================
# Metrics by place
# =========================
precision = [0.8235, 1.0]
recall = [0.8, 0.875]
f1 = [0.8116, 0.9333]

x = np.arange(len(places))
width = 0.25

plt.figure(figsize=(7,4))

plt.bar(x - width, precision, width, label='Precision')
plt.bar(x, recall, width, label='Recall')
plt.bar(x + width, f1, width, label='F1-score')

plt.xticks(x, places)
plt.xlabel("Place")
plt.ylabel("Score")
plt.ylim(0, 1.05)
plt.title("Anomaly Detection Performance")
plt.legend()

plt.grid(axis='y', linestyle='--', alpha=0.5)

plt.savefig("metrics_by_place.png", dpi=300, bbox_inches='tight')
plt.close()


# =========================
# Confusion matrix
# =========================

# place 00
cm_00 = np.array([
    [48, 6],   # TN FP
    [7, 28]    # FN TP
])

disp = ConfusionMatrixDisplay(cm_00, display_labels=["Normal","Abnormal"])
disp.plot(cmap="Blues")
plt.title("Confusion Matrix (Place 00)")
plt.savefig("confusion_matrix_place00.png", dpi=300, bbox_inches='tight')
plt.close()


# place 02
cm_02 = np.array([
    [8, 0],
    [1, 7]
])

disp = ConfusionMatrixDisplay(cm_02, display_labels=["Normal","Abnormal"])
disp.plot(cmap="Blues")
plt.title("Confusion Matrix (Place 02)")
plt.savefig("confusion_matrix_place02.png", dpi=300, bbox_inches='tight')
plt.close()


print("All graphs saved.")