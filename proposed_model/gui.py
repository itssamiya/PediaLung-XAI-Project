import os
import customtkinter as ctk
from tkinter import filedialog
from PIL import Image

from inference import LungSoundPredictor

ctk.set_appearance_mode("light")
ctk.set_default_color_theme("blue")


class PediaLungGUI(ctk.CTk):

    def __init__(self):
        super().__init__()

        self.title("PediaLung-XAI")
        self.geometry("1200x800")

        self.selected_file = None

        print("Loading AI model...")

        self.predictor = LungSoundPredictor()

        print("Model Ready.")

        # -----------------------------
        # Title
        # -----------------------------

        title = ctk.CTkLabel(
            self,
            text="PediaLung-XAI\nPediatric Respiratory Sound Classification System",
            font=("Arial", 24, "bold"),
        )

        title.pack(pady=15)

        # -----------------------------
        # Upload
        # -----------------------------

        self.upload_button = ctk.CTkButton(
            self,
            text="Upload Respiratory Sound (.wav)",
            command=self.select_file,
            width=260,
            height=40,
        )

        self.upload_button.pack(pady=10)

        # -----------------------------
        # Selected filename
        # -----------------------------

        self.file_label = ctk.CTkLabel(
            self,
            text="No file selected",
            font=("Arial", 15),
        )

        self.file_label.pack(pady=5)

        # -----------------------------
        # Predict button
        # -----------------------------

        self.predict_button = ctk.CTkButton(
            self,
            text="Predict",
            width=200,
            height=40,
            state="disabled",
            command=self.predict,
        )

        self.predict_button.pack(pady=10)

        # -----------------------------
        # Results
        # -----------------------------

        self.result_label = ctk.CTkLabel(
            self,
            text="Prediction: -",
            font=("Arial", 20, "bold"),
        )

        self.result_label.pack(pady=10)

        self.confidence_label = ctk.CTkLabel(
            self,
            text="Confidence: -",
            font=("Arial", 18),
        )

        self.confidence_label.pack()

        # -----------------------------
        # Top-3 Predictions
        # -----------------------------

        self.top3_label = ctk.CTkLabel(
            self,
            text="Top-3 Predictions",
            font=("Arial", 18, "bold"),
        )

        self.top3_label.pack(pady=(20, 5))

        self.top3_text = ctk.CTkTextbox(
            self,
            width=400,
            height=120,
        )

        self.top3_text.pack()

        # -----------------------------
        # Performance
        # -----------------------------

        self.performance = ctk.CTkLabel(
            self,
            text="Performance",
            font=("Arial", 18, "bold"),
        )

        self.performance.pack(pady=(20, 5))

        self.performance_text = ctk.CTkTextbox(
            self,
            width=400,
            height=120,
        )

        self.performance_text.pack()

        # -----------------------------
        # GradCAM
        # -----------------------------

        self.image_label = ctk.CTkLabel(
            self,
            text="Grad-CAM will appear here",
        )

        self.image_label.pack(pady=20)

    def select_file(self):

        file = filedialog.askopenfilename(filetypes=[("WAV files", "*.wav")])

        if file:

            self.selected_file = file

            self.file_label.configure(text=os.path.basename(file))

            self.predict_button.configure(state="normal")

    def predict(self):

        if self.selected_file is None:
            return

        (
            prediction,
            confidence,
            gradcam_path,
            top3,
            total_time,
            pre_time,
            infer_time,
            grad_time,
        ) = self.predictor.predict(self.selected_file)

        # --------------------------
        # Prediction
        # --------------------------

        self.result_label.configure(text=f"Prediction : {prediction}")

        self.confidence_label.configure(text=f"Confidence : {confidence*100:.2f}%")

        # --------------------------
        # Top-3
        # --------------------------

        self.top3_text.delete("1.0", "end")

        for cls, prob in top3:

            self.top3_text.insert(
                "end",
                f"{cls:<20} {prob*100:.2f}%\n",
            )

        # --------------------------
        # Performance
        # --------------------------

        self.performance_text.delete("1.0", "end")

        self.performance_text.insert("end", f"Preprocessing : {pre_time:.2f} ms\n")

        self.performance_text.insert("end", f"Inference     : {infer_time:.2f} ms\n")

        self.performance_text.insert("end", f"Grad-CAM      : {grad_time:.2f} ms\n")

        self.performance_text.insert("end", f"Total         : {total_time:.2f} ms\n")

        # --------------------------
        # GradCAM Image
        # --------------------------

        image = Image.open(gradcam_path)

        image = image.resize((420, 320))

        image = ctk.CTkImage(
            light_image=image,
            dark_image=image,
            size=(420, 320),
        )

        self.image_label.configure(
            image=image,
            text="",
        )

        self.image_label.image = image


if __name__ == "__main__":

    app = PediaLungGUI()

    app.mainloop()
