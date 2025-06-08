import os
import random
from uuid import uuid4
from fastapi import File, UploadFile, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import numpy as np
import cv2
from io import BytesIO
import time

class Modul4:
    def __init__(self, templates):
        self.templates = templates
        self._ensure_directories()
    
    def _ensure_directories(self):
        """Pastikan direktori yang diperlukan ada"""
        directories = [
            "static/uploads",
            "static/dataset", 
            "static/processed_dataset"
        ]
        for directory in directories:
            if not os.path.exists(directory):
                os.makedirs(directory)
    
    async def home(self, request: Request):
        """Halaman utama modul 4"""
        return self.templates.TemplateResponse("modules/modul4/home.html", {"request": request})
    
    # Utility functions
    def read_image_from_bytes(self, image_data):
        """Membaca gambar dari bytes"""
        np_array = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
        return img
    
    def save_image(self, image, prefix):
        """Simpan gambar ke static/uploads"""
        filename = f"modul4_{prefix}_{uuid4()}.png"
        path = os.path.join("static/uploads", filename)
        cv2.imwrite(path, image)
        return f"/static/uploads/{filename}"
    
    def detect_faces(self, image):
        """Mendeteksi wajah dalam gambar menggunakan Haar Cascades"""
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=5, 
            minSize=(30, 30)
        )
        
        # Detect eyes for better accuracy
        eyes = eye_cascade.detectMultiScale(gray)
        
        return faces, eyes
    
    # Face Detection
    async def detect_faces_upload(self, request: Request, file: UploadFile = File(...)):
        """Deteksi wajah dari gambar yang di-upload"""
        image_data = await file.read()
        img = self.read_image_from_bytes(image_data)
        
        if img is None:
            return HTMLResponse("Gagal membaca gambar", status_code=400)
        
        # Detect faces and eyes
        faces, eyes = self.detect_faces(img)
        result_img = img.copy()
        
        # Draw rectangles around faces (green)
        for (x, y, w, h) in faces:
            cv2.rectangle(result_img, (x, y), (x+w, y+h), (0, 255, 0), 3)
            cv2.putText(result_img, 'Face', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        # Draw rectangles around eyes (blue)
        for (x, y, w, h) in eyes:
            cv2.rectangle(result_img, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(result_img, 'Eye', (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        
        original_path = self.save_image(img, "original")
        result_path = self.save_image(result_img, "face_detected")
        
        return self.templates.TemplateResponse("modules/modul4/result.html", {
            "request": request,
            "original_image_path": original_path,
            "result_image_path": result_path,
            "operation": "Face & Eye Detection",
            "description": f"Ditemukan {len(faces)} wajah dan {len(eyes)} mata dalam gambar",
            "faces_count": len(faces),
            "eyes_count": len(eyes),
            "faces_coords": faces.tolist() if len(faces) > 0 else [],
            "eyes_coords": eyes.tolist() if len(eyes) > 0 else []
        })
    
    # Salt and Pepper Noise
    def add_salt_and_pepper_noise(self, image, salt_prob=0.02, pepper_prob=0.02):
        """Menambahkan noise salt and pepper ke gambar"""
        noisy_image = np.copy(image)
        total_pixels = image.size
        
        num_salt = int(total_pixels * salt_prob)
        num_pepper = int(total_pixels * pepper_prob)
        
        # Add salt noise (white pixels)
        for _ in range(num_salt):
            if len(image.shape) == 3:  # Color image
                x = random.randint(0, image.shape[1] - 1)
                y = random.randint(0, image.shape[0] - 1)
                noisy_image[y, x] = [255, 255, 255]  # White
            else:  # Grayscale image
                x = random.randint(0, image.shape[1] - 1)
                y = random.randint(0, image.shape[0] - 1)
                noisy_image[y, x] = 255
        
        # Add pepper noise (black pixels)
        for _ in range(num_pepper):
            if len(image.shape) == 3:  # Color image
                x = random.randint(0, image.shape[1] - 1)
                y = random.randint(0, image.shape[0] - 1)
                noisy_image[y, x] = [0, 0, 0]  # Black
            else:  # Grayscale image
                x = random.randint(0, image.shape[1] - 1)
                y = random.randint(0, image.shape[0] - 1)
                noisy_image[y, x] = 0
        
        return noisy_image
    
    async def add_noise(self, request: Request, file: UploadFile = File(...), 
                       salt_prob: float = Form(0.02), pepper_prob: float = Form(0.02)):
        """Menambahkan salt and pepper noise"""
        image_data = await file.read()
        img = self.read_image_from_bytes(image_data)
        
        if img is None:
            return HTMLResponse("Gagal membaca gambar", status_code=400)
        
        # Add noise
        noisy_img = self.add_salt_and_pepper_noise(img, salt_prob, pepper_prob)
        
        original_path = self.save_image(img, "original")
        result_path = self.save_image(noisy_img, "noisy")
        
        return self.templates.TemplateResponse("modules/modul4/result.html", {
            "request": request,
            "original_image_path": original_path,
            "result_image_path": result_path,
            "operation": "Salt & Pepper Noise Addition",
            "description": f"Noise ditambahkan dengan probabilitas salt: {salt_prob:.3f}, pepper: {pepper_prob:.3f}"
        })
    
    # Noise Removal
    def remove_noise_median(self, image, kernel_size=3):
        """Menghilangkan noise menggunakan median filter"""
        return cv2.medianBlur(image, kernel_size)
    
    def remove_noise_bilateral(self, image):
        """Menghilangkan noise menggunakan bilateral filter"""
        return cv2.bilateralFilter(image, 9, 75, 75)
    
    async def remove_noise(self, request: Request, file: UploadFile = File(...), 
                          method: str = Form("median"), kernel_size: int = Form(3)):
        """Menghilangkan noise dengan berbagai metode"""
        image_data = await file.read()
        img = self.read_image_from_bytes(image_data)
        
        if img is None:
            return HTMLResponse("Gagal membaca gambar", status_code=400)
        
        # Remove noise based on method
        if method == "median":
            denoised_img = self.remove_noise_median(img, kernel_size)
            description = f"Noise dihilangkan menggunakan median filter dengan kernel size {kernel_size}x{kernel_size}"
        elif method == "bilateral":
            denoised_img = self.remove_noise_bilateral(img)
            description = "Noise dihilangkan menggunakan bilateral filter untuk preservasi edge"
        elif method == "gaussian":
            denoised_img = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
            description = f"Noise dihilangkan menggunakan Gaussian blur dengan kernel size {kernel_size}x{kernel_size}"
        else:
            return HTMLResponse("Metode tidak valid", status_code=400)
        
        original_path = self.save_image(img, "original")
        result_path = self.save_image(denoised_img, f"denoised_{method}")
        
        return self.templates.TemplateResponse("modules/modul4/result.html", {
            "request": request,
            "original_image_path": original_path,
            "result_image_path": result_path,
            "operation": f"Noise Removal - {method.title()}",
            "description": description
        })
    
    # Image Sharpening
    def sharpen_image(self, image, method="kernel"):
        """Mempertajam gambar menggunakan berbagai metode"""
        if method == "kernel":
            kernel = np.array([[0, -1, 0], 
                              [-1, 5, -1], 
                              [0, -1, 0]])
            return cv2.filter2D(image, -1, kernel)
        elif method == "unsharp":
            # Unsharp masking
            gaussian = cv2.GaussianBlur(image, (9, 9), 10.0)
            return cv2.addWeighted(image, 1.5, gaussian, -0.5, 0)
        elif method == "laplacian":
            # Laplacian sharpening
            laplacian = cv2.Laplacian(image, cv2.CV_64F)
            return cv2.convertScaleAbs(laplacian)
    
    async def sharpen(self, request: Request, file: UploadFile = File(...), 
                     method: str = Form("kernel")):
        """Mempertajam gambar"""
        image_data = await file.read()
        img = self.read_image_from_bytes(image_data)
        
        if img is None:
            return HTMLResponse("Gagal membaca gambar", status_code=400)
        
        # Sharpen image
        sharpened_img = self.sharpen_image(img, method)
        
        descriptions = {
            "kernel": "Gambar dipertajam menggunakan sharpening kernel standar",
            "unsharp": "Gambar dipertajam menggunakan unsharp masking technique", 
            "laplacian": "Gambar dipertajam menggunakan Laplacian operator untuk edge enhancement"
        }
        
        original_path = self.save_image(img, "original")
        result_path = self.save_image(sharpened_img, f"sharpened_{method}")
        
        return self.templates.TemplateResponse("modules/modul4/result.html", {
            "request": request,
            "original_image_path": original_path,
            "result_image_path": result_path,
            "operation": f"Image Sharpening - {method.title()}",
            "description": descriptions.get(method, "Gambar dipertajam")
        })
    
    # Advanced Convolution
    async def advanced_convolution(self, request: Request, file: UploadFile = File(...), 
                                 operation: str = Form("blur")):
        """Operasi konvolusi lanjutan"""
        image_data = await file.read()
        img = self.read_image_from_bytes(image_data)
        
        if img is None:
            return HTMLResponse("Gagal membaca gambar", status_code=400)
        
        if operation == "blur":
            result_img = cv2.GaussianBlur(img, (15, 15), 0)
            description = "Gaussian Blur untuk efek menghaluskan gambar"
        elif operation == "emboss":
            kernel = np.array([[-2, -1, 0], 
                              [-1, 1, 1], 
                              [0, 1, 2]])
            result_img = cv2.filter2D(img, -1, kernel)
            description = "Emboss effect untuk memberikan efek timbul pada gambar"
        elif operation == "outline":
            kernel = np.array([[-1, -1, -1], 
                              [-1, 8, -1], 
                              [-1, -1, -1]])
            result_img = cv2.filter2D(img, -1, kernel)
            description = "Outline detection untuk mendeteksi garis tepi objek"
        elif operation == "motion_blur":
            # Motion blur kernel
            kernel_size = 15
            kernel = np.zeros((kernel_size, kernel_size))
            kernel[int((kernel_size-1)/2), :] = np.ones(kernel_size)
            kernel = kernel / kernel_size
            result_img = cv2.filter2D(img, -1, kernel)
            description = "Motion blur untuk efek pergerakan horizontal"
        elif operation == "edge_enhance":
            # Edge enhancement
            kernel = np.array([[-1, -1, -1],
                              [-1, 9, -1],
                              [-1, -1, -1]])
            result_img = cv2.filter2D(img, -1, kernel)
            description = "Edge enhancement untuk mempertegas garis tepi"
        else:
            return HTMLResponse("Operasi tidak valid", status_code=400)
        
        original_path = self.save_image(img, "original")
        result_path = self.save_image(result_img, f"advanced_{operation}")
        
        return self.templates.TemplateResponse("modules/modul4/result.html", {
            "request": request,
            "original_image_path": original_path,
            "result_image_path": result_path,
            "operation": f"Advanced Convolution - {operation.title().replace('_', ' ')}",
            "description": description
        })
    
    # Dataset Management (Simplified)
    async def add_to_dataset(self, request: Request, name: str = Form(...), 
                           file: UploadFile = File(...)):
        """Menambahkan wajah ke dataset"""
        image_data = await file.read()
        img = self.read_image_from_bytes(image_data)
        
        if img is None:
            return HTMLResponse("Gagal membaca gambar", status_code=400)
        
        # Detect faces
        faces, eyes = self.detect_faces(img)
        
        if len(faces) == 0:
            return HTMLResponse("Tidak ada wajah terdeteksi dalam gambar", status_code=400)
        
        # Create directory for person
        person_dir = os.path.join("static/dataset", name)
        if not os.path.exists(person_dir):
            os.makedirs(person_dir)
        
        # Save face images
        saved_faces = []
        for i, (x, y, w, h) in enumerate(faces):
            face = img[y:y+h, x:x+w]
            filename = f"{name}_{int(time.time())}_{i}.jpg"
            filepath = os.path.join(person_dir, filename)
            cv2.imwrite(filepath, face)
            saved_faces.append(f"/static/dataset/{name}/{filename}")
        
        # Also save original with detected faces
        result_img = img.copy()
        for (x, y, w, h) in faces:
            cv2.rectangle(result_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        original_path = self.save_image(img, "original")
        result_path = self.save_image(result_img, "faces_extracted")
        
        return self.templates.TemplateResponse("modules/modul4/result.html", {
            "request": request,
            "original_image_path": original_path,
            "result_image_path": result_path,
            "operation": f"Dataset Addition - {name}",
            "description": f"{len(faces)} wajah berhasil ditambahkan ke dataset untuk {name}",
            "saved_faces": saved_faces
        })