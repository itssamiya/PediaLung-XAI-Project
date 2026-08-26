import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models


import sys

# Add Graphviz binary directory directly to PATH
os.environ["PATH"] += os.pathsep + r"C:\Program Files\Graphviz\bin"

# Force CPU execution for quick testing
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

print(f"TensorFlow Version: {tf.__version__}")


def build_pedialung_tf_graph(input_shape=(128, 259, 3), num_classes=7):
    """
    Builds the computational graph in TensorFlow matching
    3-channel (MFCC + Mel + Chroma) spatial input.
    """
    # 1. Define Input Node
    inputs = layers.Input(shape=input_shape, name="Acoustic_3Channel_Input")

    # 2. Instantiate EfficientNet-B0 Backbone Node
    base_model = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights=None,  # Set to 'imagenet' if fine-tuning
        input_tensor=inputs,
        pooling="avg",
    )

    # 3. Dense Classifier Head
    x = layers.Dropout(0.2, name="Dropout_Regularization")(base_model.output)
    outputs = layers.Dense(
        num_classes, activation="softmax", name="Class_Probabilities"
    )(x)

    # Compile into computational graph
    model = models.Model(inputs=inputs, outputs=outputs, name="PediaLung_TF_Graph")
    return model


if __name__ == "__main__":
    print("\n--- Constructing TensorFlow Computational Graph ---")
    tf_model = build_pedialung_tf_graph()

    # Display layer graph summary in terminal
    tf_model.summary()

    # Test dummy inference (Simulating batch size of 1 with 128x259x3 tensor)
    dummy_input = np.random.randn(1, 128, 259, 3).astype(np.float32)
    predictions = tf_model.predict(dummy_input, verbose=0)

    print("\n--- Test Forward Pass Output ---")
    print(f"Input Shape  : {dummy_input.shape}")
    print(f"Output Shape : {predictions.shape}")
    print(f"Sample Probabilities (Sum to 1.0): \n{predictions[0]}")

    # Export graph visual image safely
    try:
        # 1. Save as PDF (Vector graphic - eliminates memory/cairo bitmap limits)
        tf.keras.utils.plot_model(
            tf_model,
            to_file="tf_architecture_graph.pdf",
            show_shapes=True,
            show_layer_names=False,
            expand_nested=False,  # Keeps the EfficientNet backbone collapsed into a single block
        )
        print(
            "\n[Success] Vector Architecture Diagram saved to: tf_architecture_graph.pdf"
        )

        # 2. Save a simplified PNG
        tf.keras.utils.plot_model(
            tf_model,
            to_file="tf_architecture_graph.png",
            show_shapes=True,
            show_layer_names=False,
            expand_nested=False,
            dpi=96,  # Lower DPI prevents Cairo surface allocation crashes
        )
        print("[Success] High-level PNG Diagram saved to: tf_architecture_graph.png")

    except Exception as e:
        print(f"\n[Error Rendering Diagram]: {e}")
