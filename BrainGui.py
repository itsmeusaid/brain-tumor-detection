import tkinter as tk
from tkinter import filedialog, Label, Button, Entry
from PIL import Image, ImageTk
import numpy as np
import time
import threading
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# ========== Parameters ==========
img_width, img_height = 224, 224
model_path = r"D:\Intern base\Project\my_brain_tumor_mobilenetv2.h5"
class_labels = ['glioma', 'meningioma', 'notumor', 'pituitary']

# ========== Load CNN Model ==========
loaded_model = load_model(model_path)

# ========== NLP Sample Training ==========
symptoms_texts = [
    "headache dizziness blurred vision", "severe headache memory loss confusion",
    "nausea seizures vision problem", "seizures vomiting and nausea",
    "hormone issues weight gain fatigue", "growth problems infertility hormonal",
    "no headache no tumor normal", "healthy normal no symptoms",
]
symptoms_labels = [
    "glioma", "glioma",
    "meningioma", "meningioma",
    "pituitary", "pituitary",
    "notumor", "notumor"
]

tfidf = TfidfVectorizer()
X_train = tfidf.fit_transform(symptoms_texts)
text_clf = LogisticRegression(max_iter=200)
text_clf.fit(X_train, symptoms_labels)

# ========== Helper Functions ==========
def preprocess_image(image_path):
    img = Image.open(image_path).resize((img_width, img_height)).convert('RGB')
    img_array = np.array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

def analyze_symptoms(text):
    if not text.strip():
        return "⚠️ Please enter some symptoms text."
    X_test = tfidf.transform([text.lower()])
    pred = text_clf.predict(X_test)[0]
    return f"Highly Chances of {pred.capitalize()}"

# ========== GUI Class ==========
class TumorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🧠 Brain Tumor Classifier")
        self.root.geometry("680x700")
        self.root.configure(bg="#5b79a5")


        self.style_font = ("Segoe UI", 12)

        # ----- Title -----
        title = Label(root, text="Brain Tumor Detection System", font=("Segoe UI", 18, "bold"), bg="#f7f7f7", fg="#333")
        title.pack(pady=20)

        # ----- Image Display -----
        self.panel = Label(root, bg="#f7f7f7")
        self.panel.pack(pady=10)

        # ----- Browse Button -----
        self.browse_btn = Button(root, text="📂 Browse MRI Image", command=self.start_browse,
                                 font=self.style_font, bg="#1976d2", fg="white", padx=10, pady=5, bd=0)
        self.browse_btn.pack(pady=8)

        # ----- Loading Label -----
        self.loading_label = Label(root, text="", font=self.style_font, bg="#f7f7f7", fg="#777")
        self.loading_label.pack(pady=4)

        # ----- CNN Result -----
        self.result_label = Label(root, text="", font=("Segoe UI", 14), bg="#f7f7f7", fg="#000")
        self.result_label.pack(pady=10)

        self.back_btn = Button(root, text="🔙 Clear Image", font=("Segoe UI", 11),
                               command=self.reset_ui, bg="#e0e0e0", bd=0)
        self.back_btn.pack_forget()

        # ----- NLP Section -----
        Label(root, text="Optional: Enter symptoms", font=("Segoe UI", 12, "italic"),
              bg="#f7f7f7", fg="#444").pack(pady=10)

        self.sym_entry = Entry(root, width=50, font=("Segoe UI", 12), bd=1, relief="solid")
        self.sym_entry.pack(ipady=6, padx=20)

        self.nlp_btn = Button(root, text="🔍 Analyze Symptoms", command=self.nlp_action,
                              font=self.style_font, bg="#4caf50", fg="white", bd=0, padx=8, pady=5)
        self.nlp_btn.pack(pady=10)

        self.nlp_label = Label(root, text="", font=("Segoe UI", 12), fg="#1a237e", bg="#f7f7f7", justify="left")
        self.nlp_label.pack(pady=5)

        # ----- Footer -----
        self.footer = Label(root, text="Status: Ready", font=("Segoe UI", 10), bg="#f7f7f7", fg="#888")
        self.footer.pack(side="bottom", pady=10)

    def reset_ui(self):
        self.panel.config(image="")
        self.panel.image = None
        self.result_label.config(text="")
        self.loading_label.config(text="")
        self.back_btn.pack_forget()
        self.browse_btn.pack(pady=8)
        self.footer.config(text="Status: Ready")

    def start_browse(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")])
        if file_path:
            self.loading_label.config(text="⏳ Processing... Please wait.")
            self.result_label.config(text="")
            self.browse_btn.pack_forget()
            self.panel.config(image="")
            self.panel.image = None
            self.root.update()
            threading.Thread(target=self.predict_image, args=(file_path,)).start()

    def predict_image(self, file_path):
        time.sleep(1.0)
        x = preprocess_image(file_path)
        pred = loaded_model.predict(x)
        idx = np.argmax(pred)
        confidence = float(pred[0][idx]) * 100

        img = Image.open(file_path).resize((img_width, img_height))
        tk_img = ImageTk.PhotoImage(img)

        def show_result():
            self.panel.config(image=tk_img)
            self.panel.image = tk_img
            self.result_label.config(
                text=f'CNN Prediction: {class_labels[idx].capitalize()} ({confidence:.2f}%)'
            )
            self.loading_label.config(text="")
            self.back_btn.pack(pady=6)
            self.footer.config(text="Status: Prediction complete")

        self.root.after(10, show_result)

    def nlp_action(self):
        user_sym = self.sym_entry.get()
        result = analyze_symptoms(user_sym)
        self.nlp_label.config(text=result)
        self.footer.config(text="Status: NLP analysis complete")

# ========== Launch App ==========
if __name__ == "__main__":
    root = tk.Tk()
    app = TumorApp(root)
    root.mainloop()