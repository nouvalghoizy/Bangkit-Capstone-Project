import cv2
import os
import logging
from typing import Dict, Tuple, Optional
import time

class FaceRecognitionSystem:
    def __init__(self):
        """Initialize the face recognition system"""
        self.setup_logging()
        self.load_components()
        self.load_user_info()
        
    def setup_logging(self):
        """Configure logging system"""
        logging.basicConfig(
            filename='face_recognition.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        
    def load_components(self):
        """Load and initialize system components"""
        try:
            # Initialize face detector
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.face_detector = cv2.CascadeClassifier(cascade_path)
            
            if self.face_detector.empty():
                raise Exception("Tidak dapat memuat cascade classifier")
            
            # Initialize face recognizer
            self.recognizer = cv2.face.LBPHFaceRecognizer_create()
            
            # Check and load training file
            training_file = 'Dataset/training.xml'
            if not os.path.exists(training_file):
                raise Exception(
                    "File training.xml tidak ditemukan! "
                    "Pastikan Anda sudah menjalankan training terlebih dahulu."
                )
            
            self.recognizer.read(training_file)
            logging.info("Successfully loaded face recognition components")
            
        except Exception as e:
            logging.error(f"Error loading components: {str(e)}")
            raise
            
    def load_user_info(self) -> None:
        """Load user information from userinfo.txt"""
        self.id_to_name = {}
        try:
            if not os.path.exists('userinfo.txt'):
                logging.warning("File userinfo.txt tidak ditemukan")
                return
                
            with open('userinfo.txt', 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            user_id, name = line.split(',')
                            self.id_to_name[int(user_id)] = name
                        except ValueError:
                            logging.warning(f"Invalid line in userinfo.txt: {line}")
                            
            logging.info(f"Loaded {len(self.id_to_name)} user records")
            
        except Exception as e:
            logging.error(f"Error loading user info: {str(e)}")
            self.id_to_name = {}
            
    def init_camera(self) -> Optional[cv2.VideoCapture]:
        """Initialize the camera"""
        try:
            video = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            if not video.isOpened():
                raise Exception("Tidak dapat membuka kamera")
            
            # Set camera properties for better quality
            video.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            video.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            video.set(cv2.CAP_PROP_FPS, 30)
            
            return video
            
        except Exception as e:
            logging.error(f"Camera initialization error: {str(e)}")
            return None
            
    def process_frame(self, frame) -> Tuple[cv2.Mat, int]:
        """Process a single frame for face detection and recognition"""
        faces_found = 0
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = self.face_detector.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )
            
            # Process each detected face
            for (x, y, w, h) in faces:
                faces_found += 1
                
                # Draw rectangle around face
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                # Predict face
                face_roi = gray[y:y+h, x:x+w]
                id_pred, confidence = self.recognizer.predict(face_roi)
                
                # Get name and confidence
                name = self.id_to_name.get(id_pred, "Unknown")
                conf_percentage = round(100 - confidence, 1)
                
                # Display name and confidence
                label = f"{name} ({conf_percentage}%)"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.7, 1)[0]
                
                # Draw background rectangle for text
                cv2.rectangle(frame, 
                            (x, y - label_size[1] - 10),
                            (x + label_size[0], y),
                            (0, 255, 0),
                            cv2.FILLED)
                
                # Draw text
                cv2.putText(frame, label,
                           (x, y - 5),
                           cv2.FONT_HERSHEY_DUPLEX,
                           0.7, (0, 0, 0), 1)
                
            return frame, faces_found
            
        except Exception as e:
            logging.error(f"Error processing frame: {str(e)}")
            return frame, faces_found
            
    def run(self):
        """Main function to run face recognition"""
        try:
            print("Menginisialisasi sistem pengenalan wajah...")
            
            # Initialize camera
            video = self.init_camera()
            if video is None:
                raise Exception("Tidak dapat menginisialisasi kamera")
            
            print("\nSistem pengenalan wajah aktif:")
            print("- Tekan 'q' untuk keluar")
            print("- Tekan 's' untuk screenshot")
            
            frame_count = 0
            start_time = time.time()
            screenshots_dir = "Screenshots"
            
            # Create screenshots directory if it doesn't exist
            if not os.path.exists(screenshots_dir):
                os.makedirs(screenshots_dir)
            
            while True:
                # Read frame
                ret, frame = video.read()
                if not ret:
                    raise Exception("Tidak dapat membaca frame dari kamera")
                
                # Process frame
                frame, faces_found = self.process_frame(frame)
                
                # Calculate and display FPS
                frame_count += 1
                if frame_count % 30 == 0:
                    fps = frame_count / (time.time() - start_time)
                    cv2.putText(frame, f"FPS: {fps:.1f}",
                              (10, 30),
                              cv2.FONT_HERSHEY_SIMPLEX,
                              1, (0, 255, 0), 2)
                
                # Show frame
                cv2.imshow("Face Recognition", frame)
                
                # Handle keyboard events
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    # Take screenshot
                    timestamp = time.strftime("%Y%m%d-%H%M%S")
                    filename = f"{screenshots_dir}/screenshot_{timestamp}.jpg"
                    cv2.imwrite(filename, frame)
                    print(f"Screenshot saved: {filename}")
                    
        except Exception as e:
            logging.error(f"Error in main loop: {str(e)}")
            print(f"\nError: {str(e)}")
            
        finally:
            if 'video' in locals():
                video.release()
            cv2.destroyAllWindows()
            print("\nProgram selesai.")

def main():
    """Main entry point"""
    try:
        system = FaceRecognitionSystem()
        system.run()
    except Exception as e:
        print(f"\nError fatal: {str(e)}")
        print("Silakan cek file log untuk detail lebih lanjut.")
        
if __name__ == "__main__":
    main()