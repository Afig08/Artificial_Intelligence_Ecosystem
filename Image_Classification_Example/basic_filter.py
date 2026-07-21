from PIL import Image, ImageFilter, ImageDraw
import matplotlib.pyplot as plt
import os
import random


def apply_blur_filter(
    image_path,
    output_path="blurred_image.png"
):
    """
    Applies a Gaussian blur to an image.
    """

    try:
        img = Image.open(image_path)
        img_resized = img.resize((128, 128))

        img_blurred = img_resized.filter(
            ImageFilter.GaussianBlur(radius=2)
        )

        plt.imshow(img_blurred)
        plt.axis("off")

        plt.savefig(
            output_path,
            bbox_inches="tight",
            pad_inches=0
        )

        plt.close()

        print(
            f"Processed image saved as '{output_path}'."
        )

    except Exception as e:
        print(f"Error processing image: {e}")


def apply_spaghetti_filter(
    image_path,
    output_path="spaghetti_monster.png",
    noodle_count=50,
    meatball_count=10
):
    """
    Transforms the image into a spaghetti monster by
    drawing noodle strands and meatball-like circles.
    """

    try:
        img = Image.open(image_path).convert("RGBA")
        img_resized = img.resize((256, 256))

        overlay = Image.new(
            "RGBA",
            img_resized.size,
            (0, 0, 0, 0)
        )

        draw = ImageDraw.Draw(overlay)

        # Draw noodles
        for _ in range(noodle_count):
            color = (
                random.randint(200, 255),
                random.randint(180, 230),
                random.randint(50, 100),
                180
            )

            x0 = random.randint(0, 255)
            y0 = random.randint(0, 255)

            x1 = random.randint(0, 255)
            y1 = random.randint(0, 255)

            points = [(x0, y0)]

            for t in range(1, 5):
                xt = (
                    x0
                    + (x1 - x0) * t / 5
                    + random.randint(-20, 20)
                )

                yt = (
                    y0
                    + (y1 - y0) * t / 5
                    + random.randint(-20, 20)
                )

                points.append((xt, yt))

            points.append((x1, y1))

            draw.line(
                points,
                fill=color,
                width=random.randint(5, 12)
            )

        # Draw meatballs
        for _ in range(meatball_count):
            cx = random.randint(0, 255)
            cy = random.randint(0, 255)

            radius = random.randint(10, 25)

            bounding_box = [
                cx - radius,
                cy - radius,
                cx + radius,
                cy + radius
            ]

            color = (
                random.randint(80, 120),
                random.randint(40, 60),
                random.randint(20, 30),
                200
            )

            draw.ellipse(
                bounding_box,
                fill=color
            )

        combined = Image.alpha_composite(
            img_resized,
            overlay
        ).convert("RGB")

        combined.save(output_path)

        print(
            f"Spaghetti monster image saved as "
            f"'{output_path}'."
        )

    except Exception as e:
        print(f"Error processing image: {e}")


def apply_pokemon_filter(
    image_path,
    output_path="pokemon_style.png"
):
    """
    Applies a bright, colorful, cel-shaded cartoon effect
    inspired by monster-training animation.
    """

    try:
        # Open the image and convert it to RGB
        img = Image.open(image_path).convert("RGB")

        # Resize the image to a consistent size
        img_resized = img.resize((512, 512))

        # Smooth the image while preserving major shapes
        smooth = img_resized.filter(
            ImageFilter.MedianFilter(size=5)
        )

        # Convert the image to HSV so saturation and
        # brightness can be adjusted separately
        hsv = smooth.convert("HSV")
        hue, saturation, brightness = hsv.split()

        # Increase color saturation
        saturation = saturation.point(
            lambda value: min(
                255,
                int(value * 1.6)
            )
        )

        # Increase brightness slightly
        brightness = brightness.point(
            lambda value: min(
                255,
                int(value * 1.15)
            )
        )

        colorful = Image.merge(
            "HSV",
            (
                hue,
                saturation,
                brightness
            )
        ).convert("RGB")

        # Reduce the number of colors to create
        # a cel-shaded cartoon appearance
        posterized = colorful.quantize(
            colors=32
        ).convert("RGB")

        # Convert the original image to grayscale
        grayscale = img_resized.convert("L")

        # Detect image edges
        edges = grayscale.filter(
            ImageFilter.FIND_EDGES
        )

        # Convert edge values into stronger black-and-white lines
        edges = edges.point(
            lambda value: (
                0 if value < 35 else 255
            )
        )

        # Invert the edges so the outlines become dark
        edges = Image.eval(
            edges,
            lambda value: 255 - value
        )

        # Convert the grayscale edge image to RGB
        edge_layer = Image.merge(
            "RGB",
            (
                edges,
                edges,
                edges
            )
        )

        # Blend the dark outlines with the colorful image
        pokemon_style = Image.blend(
            posterized,
            edge_layer,
            alpha=0.25
        )

        # Sharpen the final image
        pokemon_style = pokemon_style.filter(
            ImageFilter.SHARPEN
        )

        # Save the result
        pokemon_style.save(output_path)

        print(
            f"Pokemon-style image saved as "
            f"'{output_path}'."
        )

    except Exception as e:
        print(f"Error processing image: {e}")


if __name__ == "__main__":
    print(
        "Image Processor "
        "(type 'exit' to quit)\n"
        "Available filters: blur, spaghetti, pokemon"
    )

    while True:
        image_path = input(
            "\nEnter image filename "
            "(or 'exit' to quit): "
        ).strip()

        if image_path.lower() == "exit":
            print("Goodbye!")
            break

        if not os.path.isfile(image_path):
            print(f"File not found: {image_path}")
            continue

        choice = input(
            "Choose filter "
            "('blur', 'spaghetti', or 'pokemon'): "
        ).strip().lower()

        base, extension = os.path.splitext(
            image_path
        )

        if choice == "blur":
            output_file = (
                f"{base}_blurred{extension}"
            )

            apply_blur_filter(
                image_path,
                output_file
            )

        elif choice == "spaghetti":
            output_file = (
                f"{base}_spaghetti{extension}"
            )

            apply_spaghetti_filter(
                image_path,
                output_file
            )

        elif choice == "pokemon":
            output_file = (
                f"{base}_pokemon{extension}"
            )

            apply_pokemon_filter(
                image_path,
                output_file
            )

        else:
            print(
                "Unknown filter choice. "
                "Please select 'blur', "
                "'spaghetti', or 'pokemon'."
            )