import os
import sys
from tkinter import filedialog
import customtkinter as ctk
from PIL import Image

ctk.set_appearance_mode("light")
ctk.set_default_color_theme("blue")

# Ensure proper path resolution
base_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(base_dir, ".."))
if root_dir not in sys.path:
    sys.path.append(root_dir)

# Import the updated EfficientNet predictor engine
from proposed_model.app_inference import PediaLungAppPredictor as EfficientNetPredictor


class PediaLungGUI(ctk.CTk):

    def __init__(self):
        super().__init__()

        self.title("PediaLung-XAI")
        self.geometry("1200x850")

        self.selected_file = None

        print("Loading AI model...")
        # FIX 1: Instantiate using the imported EfficientNetPredictor class
        self.predictor = EfficientNetPredictor()
        print("Model Ready.")

        # -----------------------------
        # Title
        # -----------------------------
        title = ctk.CTkLabel(
            self,
            text="PediaLung-XAI\nPediatric Respiratory Sound Classification System",
            font=("Arial", 22, "bold"),
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
            font=("Arial", 14),
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
        # Results Section Frame
        # -----------------------------
        self.results_frame = ctk.CTkFrame(self)
        self.results_frame.pack(fill="both", expand=True, padx=20, pady=10)

        # Left Column: Metrics & Outputs
        self.left_panel = ctk.CTkFrame(self.results_frame, fg_color="transparent")
        self.left_panel.pack(side="left", fill="both", expand=True, padx=10, pady=10)

        self.result_label = ctk.CTkLabel(
            self.left_panel,
            text="Prediction: -",
            font=("Arial", 18, "bold"),
        )
        self.result_label.pack(anchor="w", pady=5)

        self.confidence_label = ctk.CTkLabel(
            self.left_panel,
            text="Confidence: -",
            font=("Arial", 16),
        )
        self.confidence_label.pack(anchor="w", pady=5)

        self.top3_label = ctk.CTkLabel(
            self.left_panel,
            text="Top-3 Predictions",
            font=("Arial", 16, "bold"),
        )
        self.top3_label.pack(anchor="w", pady=(15, 5))

        self.top3_text = ctk.CTkTextbox(
            self.left_panel,
            width=380,
            height=100,
        )
        self.top3_text.pack(anchor="w")

        self.performance = ctk.CTkLabel(
            self.left_panel,
            text="Performance",
            font=("Arial", 16, "bold"),
        )
        self.performance.pack(anchor="w", pady=(15, 5))

        self.performance_text = ctk.CTkTextbox(
            self.left_panel,
            width=380,
            height=100,
        )
        self.performance_text.pack(anchor="w")

        # Right Column: GradCAM Visualization
        self.right_panel = ctk.CTkFrame(self.results_frame, fg_color="transparent")
        self.right_panel.pack(side="right", fill="both", expand=True, padx=10, pady=10)

        self.image_label = ctk.CTkLabel(
            self.right_panel,
            text="Grad-CAM overlay will appear here",
            font=("Arial", 14),
        )
        self.image_label.pack(expand=True, pady=10)

    def select_file(self):
        file = filedialog.askopenfilename(filetypes=[("WAV files", "*.wav")])
        if file:
            self.selected_file = file
            self.file_label.configure(text=os.path.basename(file))
            self.predict_button.configure(state="normal")

    def predict(self):
        if self.selected_file is None:
            return

        self.predict_button.configure(state="disabled", text="Analyzing...")
        self.update_idletasks()

        # FIX 2: Parse dictionary returned by PediaLungAppPredictor safely
        res = self.predictor.predict(self.selected_file)

        prediction = res["prediction"]
        confidence = res["confidence"]
        top3 = res["top3"]
        gradcam_path = res["gradcam_path"]
        pre_time = res["preprocessing_time"]
        infer_time = res["inference_time"]
        grad_time = res["gradcam_time"]
        total_time = res["total_time"]

        # --------------------------
        # Prediction & Confidence
        # --------------------------
        self.result_label.configure(text=f"Prediction : {prediction}")
        self.confidence_label.configure(text=f"Confidence : {confidence*100:.2f}%")

        # --------------------------
        # Top-3 Probabilities
        # --------------------------
        self.top3_text.delete("1.0", "end")
        for cls, prob in top3:
            self.top3_text.insert(
                "end",
                f"{cls:<20} {prob*100:.2f}%\n",
            )

        # --------------------------
        # Performance Latencies
        # --------------------------
        self.performance_text.delete("1.0", "end")
        self.performance_text.insert("end", f"Preprocessing : {pre_time:.2f} ms\n")
        self.performance_text.insert("end", f"Inference     : {infer_time:.2f} ms\n")
        self.performance_text.insert("end", f"Grad-CAM      : {grad_time:.2f} ms\n")
        self.performance_text.insert("end", f"Total         : {total_time:.2f} ms\n")

        # --------------------------
        # GradCAM Image Overlay
        # --------------------------
        image = Image.open(gradcam_path)
        ctk_img = ctk.CTkImage(
            light_image=image,
            dark_image=image,
            size=(520, 340),
        )

        self.image_label.configure(
            image=ctk_img,
            text="",
        )
        self.image_label.image = ctk_img
        self.predict_button.configure(state="normal", text="Predict")


if __name__ == "__main__":
    app = PediaLungGUI()
    app.mainloop()
