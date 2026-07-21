import tensorflow as tf
import numpy as np
import cv2

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import (
    preprocess_input,
    decode_predictions
)
from tensorflow.keras.preprocessing import image


# Suppress most TensorFlow logs
tf.get_logger().setLevel("ERROR")


# Load MobileNetV2 with pretrained ImageNet weights
base_model = MobileNetV2(weights="imagenet")


def make_gradcam_heatmap(
    img_array,
    model,
    last_conv_layer_name,
    pred_index=None
):
    """
    Creates a Grad-CAM heatmap showing which image regions most
    strongly influenced the selected prediction.
    """

    # Create a model that returns both convolutional feature maps
    # and the normal classification predictions
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[
            model.get_layer(last_conv_layer_name).output,
            model.output
        ]
    )

    # Record operations so TensorFlow can calculate gradients
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)

        # Use the highest-confidence class by default
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])

        class_score = predictions[:, pred_index]

    # Calculate how the class score changes relative to the
    # convolutional feature maps
    gradients = tape.gradient(
        class_score,
        conv_outputs
    )

    if gradients is None:
        raise RuntimeError(
            "TensorFlow could not calculate Grad-CAM gradients."
        )

    # Calculate one importance weight for each feature channel
    pooled_gradients = tf.reduce_mean(
        gradients,
        axis=(0, 1, 2)
    )

    # Remove the batch dimension
    conv_outputs = conv_outputs[0]

    # Combine feature maps using their importance weights
    heatmap = conv_outputs @ pooled_gradients[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Keep only positive contributions
    heatmap = tf.maximum(heatmap, 0)

    # Normalize the values between 0 and 1
    maximum = tf.reduce_max(heatmap)

    if maximum > 0:
        heatmap = heatmap / maximum

    return heatmap.numpy()


def overlay_heatmap(
    img_path,
    heatmap,
    alpha=0.4,
    colormap=cv2.COLORMAP_JET
):
    """
    Places the Grad-CAM heatmap over the original image.
    """

    img = cv2.imread(img_path)

    if img is None:
        raise FileNotFoundError(
            f"Could not load image: {img_path}"
        )

    # Resize heatmap to match the original image
    heatmap = cv2.resize(
        heatmap,
        (img.shape[1], img.shape[0])
    )

    # Convert heatmap to an 8-bit image
    heatmap = np.uint8(255 * heatmap)

    # Apply a color map
    heatmap_color = cv2.applyColorMap(
        heatmap,
        colormap
    )

    # Blend the original image with the colored heatmap
    overlay = cv2.addWeighted(
        img,
        1 - alpha,
        heatmap_color,
        alpha,
        0
    )

    return overlay


def classify_and_gradcam(image_path, top=3):
    """
    Classifies an image, prints its top predictions, and saves
    a Grad-CAM visualization.
    """

    # Load and resize the image for MobileNetV2
    img = image.load_img(
        image_path,
        target_size=(224, 224)
    )

    # Convert the image into a numerical array
    img_array = image.img_to_array(img)

    # Apply MobileNetV2 preprocessing
    img_array = preprocess_input(img_array)

    # Add a batch dimension
    img_array = np.expand_dims(
        img_array,
        axis=0
    )

    # Generate ImageNet predictions
    predictions = base_model.predict(
        img_array,
        verbose=0
    )

    decoded_predictions = decode_predictions(
        predictions,
        top=top
    )[0]

    # Create the Grad-CAM heatmap
    heatmap = make_gradcam_heatmap(
        img_array,
        base_model,
        "Conv_1"
    )

    # Overlay the heatmap on the original image
    overlay = overlay_heatmap(
        image_path,
        heatmap
    )

    # Build and save the output filename
    output_path = (
        image_path.rsplit(".", 1)[0]
        + "_gradcam.jpg"
    )

    if not cv2.imwrite(output_path, overlay):
        raise IOError(
            f"Could not save output image: {output_path}"
        )

    # Display predictions
    print(f"\nTop-{top} Predictions for {image_path}:")

    for rank, (_, label, score) in enumerate(
        decoded_predictions,
        start=1
    ):
        readable_label = label.replace("_", " ")

        print(
            f"  {rank}: {readable_label} "
            f"({score:.2%})"
        )

    print(
        f"Grad-CAM overlay saved to: {output_path}\n"
    )


if __name__ == "__main__":
    print(
        "Grad-CAM Image Classifier "
        "(type 'exit' to quit)\n"
    )

    while True:
        path = input(
            "Enter image filename: "
        ).strip()

        if path.lower() == "exit":
            print("Goodbye!")
            break

        try:
            classify_and_gradcam(path)

        except FileNotFoundError:
            print(
                f"Error: Could not find '{path}'.\n"
            )

        except Exception as error:
            print(
                f"Error processing '{path}': "
                f"{error}\n"
            )