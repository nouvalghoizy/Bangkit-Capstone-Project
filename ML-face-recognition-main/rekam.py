import cv2
import time
import os
from tkinter import *
from tkinter import messagebox
from tkinter import ttk  # Untuk styling yang lebih baik

def capture_faces():
    # Get values from input fields
    user_id = id_entry.get()
    user_name = name_entry.get()
    
    if not user_id or not user_name:
        messagebox.showerror("Error", "ID dan Nama harus diisi!")
        return
    
    try:
        # Get the current directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Create path to the cascade file
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        
        # Initialize camera
        camera = 0
        video = cv2.VideoCapture(camera, cv2.CAP_DSHOW)
        
        # Load face detection classifier
        faceDeteksi = cv2.CascadeClassifier(cascade_path)
        
        # Check if cascade loaded successfully
        if faceDeteksi.empty():
            messagebox.showerror("Error", "Couldn't load face cascade classifier!")
            return
        
        # Create Dataset directory if it doesn't exist
        dataset_dir = 'Dataset'
        if not os.path.exists(dataset_dir):
            os.makedirs(dataset_dir)
        
        # Create user info file
        with open('userinfo.txt', 'a') as f:
            f.write(f"{user_id},{user_name}\n")
        
        a = 0
        while True:
            a = a + 1
            check, frame = video.read()
            
            if not check:
                messagebox.showerror("Error", "Tidak dapat membaca kamera!")
                break
            
            # Convert to grayscale
            abu = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            wajah = faceDeteksi.detectMultiScale(abu, 1.3, 5)
            
            for (x, y, w, h) in wajah:
                # Save face image
                image_path = os.path.join(dataset_dir, f'User.{str(user_id)}.{str(a)}.jpg')
                cv2.imwrite(image_path, abu[y:y+h, x:x+w])
                
                # Draw rectangle around face
                cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
                
                # Display counter
                cv2.putText(frame, f"Gambar ke: {a}/100", (10, 100), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, f"Tekan 'q' untuk keluar", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Display the frame
            cv2.imshow("Pengambilan Data Wajah", frame)
            
            # Break if 'q' is pressed or after 30 photos
            if cv2.waitKey(100) & 0xFF == ord('q') or a >= 100:
                break
        
        # Release resources
        video.release()
        cv2.destroyAllWindows()
        
        messagebox.showinfo("Sukses", f"Berhasil mengambil {a} gambar wajah untuk {user_name}")
        
    except Exception as e:
        messagebox.showerror("Error", f"Terjadi kesalahan: {str(e)}")
        
    finally:
        # Clear the entry fields
        id_entry.delete(0, END)
        name_entry.delete(0, END)

# Create main window
root = Tk()
root.title("Pengambilan Data Wajah")

# Set window size and position
window_width = 400
window_height = 500
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
x = (screen_width - window_width) // 2
y = (screen_height - window_height) // 2
root.geometry(f"{window_width}x{window_height}+{x}+{y}")

# Configure style
root.configure(bg='#f0f0f0')
style = ttk.Style()
style.configure('TButton', padding=10, font=('Helvetica', 12))
style.configure('TLabel', background='#f0f0f0', font=('Helvetica', 12))
style.configure('TEntry', font=('Helvetica', 12))

# Create main frame
main_frame = Frame(root, bg='#f0f0f0', padx=20, pady=20)
main_frame.pack(expand=True, fill=BOTH)

# Title
title_label = Label(main_frame, text="Pengambilan Data Wajah", 
                   font=("Helvetica", 18, "bold"), 
                   bg='#f0f0f0')
title_label.pack(pady=(0, 20))

# Input frame
input_frame = Frame(main_frame, bg='#f0f0f0')
input_frame.pack(fill=X, pady=10)

# ID input
id_label = Label(input_frame, text="ID:", 
                bg='#f0f0f0', 
                font=("Helvetica", 12))
id_label.pack(anchor=W)
id_entry = Entry(input_frame, font=("Helvetica", 12))
id_entry.pack(fill=X, pady=(5, 10))

# Name input
name_label = Label(input_frame, text="Nama:", 
                  bg='#f0f0f0', 
                  font=("Helvetica", 12))
name_label.pack(anchor=W)
name_entry = Entry(input_frame, font=("Helvetica", 12))
name_entry.pack(fill=X, pady=(5, 20))

# Start button with updated style
start_button = Button(input_frame, 
                     text="Mulai Pengambilan Data",
                     command=capture_faces,
                     font=("Helvetica", 12, "bold"),
                     bg='#4CAF50',
                     fg='white',
                     activebackground='#45a049',
                     activeforeground='white',
                     relief=RAISED,
                     padx=20,
                     pady=10)
start_button.pack(pady=10)

# Instructions
instructions = """
Petunjuk Penggunaan:
1. Masukkan ID dan Nama
2. Klik tombol 'Mulai Pengambilan Data'
3. Posisikan wajah di depan kamera
4. Tunggu hingga 100 gambar terambil
5. Atau tekan 'q' untuk berhenti
"""
instruction_label = Label(main_frame, 
                        text=instructions,
                        justify=LEFT,
                        bg='#f0f0f0',
                        font=("Helvetica", 11))
instruction_label.pack(pady=20)

# Start the main loop
root.mainloop()