import kagglehub
import tensorflow as tf
from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image, ImageOps
import numpy as np
import torch
from torch.nn.functional import cross_entropy
from torch.optim import Adam

def preprocess_images(image_batch):
        """
        Preprocesses a batch of images using ViTImageProcessor.
        """
        images = [Image.fromarray((img * 255).astype(np.uint8)) for img in image_batch]
        processed_images = processor(images=images, return_tensors="pt")
        return processed_images["pixel_values"]

def train_model(train_gen, val_gen, model, epochs=1):
    optimizer = Adam(model.parameters(), lr=5e-5)
    steps_per_epoch = len(train_gen)
    validation_steps = len(val_gen)

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        model.train()
        train_loss = 0

        # Training loop
        for i in range(steps_per_epoch):
            try:
                x_batch, y_batch = next(train_gen)
                x_batch = preprocess_images(x_batch).to(device)
                y_batch = torch.tensor(y_batch, dtype=torch.long).to(device)

                optimizer.zero_grad()
                outputs = model(pixel_values=x_batch)
                loss = cross_entropy(outputs.logits, y_batch)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

                if i % 10 == 0:
                    print(f"Batch {i}/{steps_per_epoch}, Loss: {loss.item():.4f}")
                    torch.cuda.empty_cache()

            except StopIteration:
                break

        avg_train_loss = train_loss / steps_per_epoch
        print(f"Training Loss: {avg_train_loss:.4f}")

        # Validation loop
        model.eval()
        val_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for i in range(validation_steps):
                try:
                    x_batch, y_batch = next(val_gen)
                    x_batch = preprocess_images(x_batch).to(device)
                    y_batch = torch.tensor(y_batch, dtype=torch.long).to(device)

                    outputs = model(pixel_values=x_batch)
                    loss = cross_entropy(outputs.logits, y_batch)
                    val_loss += loss.item()

                    # Calculate accuracy
                    preds = torch.argmax(outputs.logits, dim=1)
                    correct += (preds == y_batch).sum().item()
                    total += y_batch.size(0)

                except StopIteration:
                    break

        avg_val_loss = val_loss / validation_steps
        val_accuracy = correct / total if total > 0 else 0
        print(f"Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

        # Reset generators at the end of each epoch
        train_gen.reset()
        val_gen.reset()

def preprocess_image(image_path, processor):
        image = Image.open(image_path).convert("RGB")  # Ensure 3 channels (RGB)
        resized_image = image.resize((224, 224), Image.Resampling.LANCZOS)  # Resize to (224, 224)

        # Optional: If grayscale effect is required, convert and revert back to RGB
        processed_image = ImageOps.grayscale(resized_image).convert("RGB")

        # Preprocess the image with the processor
        inputs = processor(images=processed_image, return_tensors="pt")["pixel_values"]
        return inputs


def predict_single_image(image_path, model, processor, inputs, class_labels=["FAKE", "REAL"]):
    model.eval()
    with torch.no_grad():
        outputs = model(inputs)
    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=1).cpu().numpy()[0]

    # Get the predicted class
    predicted_class_idx = torch.argmax(logits, dim=1).item()
    predicted_label = class_labels[predicted_class_idx]
    confidence = probabilities[predicted_class_idx]

    return predicted_label, confidence


if __name__ == "__main__":

    path = kagglehub.dataset_download("yadavadarsh55/ai-generated-and-real-images")

    print("Path to dataset files:", path)


    processor = ViTImageProcessor.from_pretrained('facebook/deit-tiny-patch16-224')
    model = ViTForImageClassification.from_pretrained('facebook/deit-tiny-patch16-224')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Define directories for training and validation data
    train_dir = "/root/.cache/kagglehub/datasets/yadavadarsh55/ai-generated-and-real-images/versions/1/data"

    # Define parameters
    batch_size = 32
    img_size = (224, 224)

    # Create ImageDataGenerators
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1.0 / 255,
        validation_split=0.2
    )

    # Create generators
    train_gen = train_datagen.flow_from_directory(
        train_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode="binary",
        subset="training",
        seed=123
    )

    val_gen = train_datagen.flow_from_directory(
        train_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode="binary",
        subset="validation",
        seed=123
    )

    train_model(train_gen, val_gen, model, epochs=3)

    model.save_pretrained("fine_tuned_vit_model")
    processor.save_pretrained("fine_tuned_vit_model")


    model_path = "fine_tuned_vit_model"
    model = ViTForImageClassification.from_pretrained(model_path)
    processor = ViTImageProcessor.from_pretrained(model_path)

    

    # Example usage
    image_path = "photo_2024-03-19_22-07-07.jpg"  # Path to the image you want to predict
    class_labels = ["FAKE", "REAL"]  # Define your class labels

    input = preprocess_image(image_path, processor)
    predicted_label, confidence = predict_single_image(image_path, model, processor, input, class_labels)

    print(f"Predicted Label: {predicted_label}")
    print(f"Confidence Score: {confidence:.2f}")