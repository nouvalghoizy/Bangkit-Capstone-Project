import os
import numpy as np
from PIL import Image
import logging
import time
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from typing import Tuple, List

class FaceTrainerTF:
    def __init__(self):
        """Initialize the face trainer with necessary components"""
        self.setup_logging()
        self.dataset_path = 'Dataset'
        self.model_save_path = 'model/saved_model'
        self.input_shape = (100, 100, 1)  # Target image size
        self.batch_size = 32
        self.epochs = 10

    def setup_logging(self):
        """Configure logging system"""
        logging.basicConfig(
            filename='face_training_tf.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

    def ensure_dataset_directory(self):
        """Ensure dataset directory exists"""
        try:
            if not os.path.exists(self.dataset_path):
                os.makedirs(self.dataset_path)
                logging.info(f"Created dataset directory: {self.dataset_path}")
            return True
        except Exception as e:
            logging.error(f"Error creating dataset directory: {str(e)}")
            return False

    def preprocess_image(self, image_path: str) -> np.ndarray:
        """Preprocess image for training"""
        image = Image.open(image_path).convert('L')  # Convert to grayscale
        image = image.resize((100, 100))  # Resize to target input shape
        return np.array(image, 'float32') / 255.0  # Normalize pixel values

    def get_training_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get face samples and IDs from dataset"""
        face_samples = []
        ids = []
        skipped_count = 0

        try:
            # Get all files in directory
            image_paths = [os.path.join(self.dataset_path, f)
                           for f in os.listdir(self.dataset_path)
                           if f.endswith(('.jpg', '.jpeg', '.png'))]

            if not image_paths:
                raise Exception("No images found in the dataset folder")

            logging.info(f"Found {len(image_paths)} images in dataset")
            print(f"Processing {len(image_paths)} images...")

            for image_path in image_paths:
                try:
                    # Extract ID from filename
                    id_str = os.path.split(image_path)[-1].split(".")[1]
                    if not id_str.isdigit():
                        raise ValueError(f"Invalid ID format in filename: {image_path}")
                    id_num = int(id_str)

                    # Preprocess image and add to dataset
                    processed_image = self.preprocess_image(image_path)
                    face_samples.append(processed_image)
                    ids.append(id_num)

                except Exception as e:
                    logging.error(f"Error processing image {image_path}: {str(e)}")
                    skipped_count += 1
                    continue

            if not face_samples:
                raise Exception("No valid faces found in the dataset")

            logging.info(f"Successfully processed {len(face_samples)} images")
            if skipped_count > 0:
                logging.warning(f"Skipped {skipped_count} images")

            return np.array(face_samples), np.array(ids)

        except Exception as e:
            logging.error(f"Error getting training data: {str(e)}")
            raise

    def build_model(self) -> models.Sequential:
        """Build the CNN model"""
        model = models.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=self.input_shape),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dense(len(set(self.y_train)), activation='softmax')
        ])
        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        return model

    def train_recognizer(self):
        """Train the face recognizer with collected data"""
        try:
            if not self.ensure_dataset_directory():
                raise Exception("Cannot create dataset directory")

            print("Starting training process...")
            start_time = time.time()

            # Get training data
            faces, ids = self.get_training_data()
            faces = faces.reshape(-1, 100, 100, 1)  # Reshape for CNN
            self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
                faces, ids, test_size=0.2, random_state=42)

            # Build and train the model
            model = self.build_model()
            model.fit(
                self.x_train, self.y_train,
                validation_data=(self.x_test, self.y_test),
                batch_size=self.batch_size,
                epochs=self.epochs
            )

            # Save the model
            os.makedirs(os.path.dirname(self.model_save_path), exist_ok=True)
            model.save(self.model_save_path)

            end_time = time.time()
            training_time = round(end_time - start_time, 2)

            success_message = (
                f"Training completed!\n"
                f"- Processed faces: {len(faces)}\n"
                f"- Unique IDs: {len(set(ids))}\n"
                f"- Training time: {training_time} seconds\n"
                f"- Model saved to: {self.model_save_path}"
            )

            print(success_message)
            logging.info(success_message.replace('\n', ' '))

        except Exception as e:
            error_message = f"Error during training: {str(e)}"
            print(f"Error: {error_message}")
            logging.error(error_message)
            raise


def main():
    """Main function to run the training process"""
    try:
        trainer = FaceTrainerTF()
        trainer.train_recognizer()
    except Exception as e:
        print(f"\nTraining failed: {str(e)}")
        print("Check the log file for more details.")


if __name__ == "__main__":
    main()
