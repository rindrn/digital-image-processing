import os
import random
from uuid import uuid4
from fastapi import File, UploadFile, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import numpy as np
import cv2

class Modul4:
    def __init__(self, templates):
        self.templates = templates
        self._ensure_directories()
    
    def _ensure_directories(self):
        for dir_name in ["static/uploads", "dataset", "processed_dataset"]:
            if not os.path.exists(dir_name):
                os.makedirs(dir_name)
    
    async def home(self, request: Request):
        return self.templates.TemplateResponse("modules/modul4/home.html", {"request": request})
    
    # Face Detection
    async def detect_faces(self, request: Request, file: UploadFile = File(...)):
        image_data = await file.read()
        np_array = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
        
        if img is None:
            return HTMLResponse("Gagal membaca gambar", status_code=400)
        
        # Load face cascade
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        # Draw rectangles around faces
        result_img = img.copy()
        for (x, y, w, h) in faces:
            cv2.rectangle(result_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(result_img, f'Face {len(faces)}', (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        original_path = self.save_image(img, "modul4_original")
        modified_path = self.save_image(result_img, "modul4_faces")
        
        return self.templates.TemplateResponse("modules/modul4/result.html", {
            "request": request,
            "original_image_path": original_path,
            "modified_image_path": modified_path,
            "operation": f"Face Detection - {len(faces)} faces found"
        })
    
    # Add Salt and Pepper Noise
    async def add_salt_pepper_noise(self, request: Request, file: UploadFile = File(...), 
                                   salt_prob: float = Form(0.02), pepper_prob: float = Form(0.02)):
        image_data = await file.read()
        np_array = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
        
        if img is None:
            return HTMLResponse("Gagal membaca gambar", status_code=400)
        
        # Add salt and pepper noise
        noisy_img = self._add_salt_pepper_noise(img, salt_prob, pepper_prob)
        
        original_path = self.save_image(img, "modul4_original")
        modified_path = self.save_image(noisy_img, "modul4_noisy")
        
        return self.templates.TemplateResponse("modules/modul4/result.html", {
            "request": request,
            "original_image_path": original_path,
            "modified_image_path": modified_path,
            "operation": f"Salt & Pepper Noise (Salt: {salt_prob}, Pepper: {pepper_prob})"
        })
    
    # Remove Noise with Median Filter
    async def remove_noise(self, request: Request, file: UploadFile = File(...), 
                          kernel_size: int = Form(5)):
        image_data = await file.read()
        np_array = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
        
        if img is None:
            return HTMLResponse("Gagal membaca gambar", status_code=400)
        
        # Apply median filter
        denoised_img = cv2.medianBlur(img, kernel_size)
        
        original_path = self.save_image(img, "modul4_original")
        modified_path = self.save_image(denoised_img, "modul4_denoised")
        
        return self.templates.TemplateResponse("modules/modul4/result.html", {
            "request": request,
            "original_image_path": original_path,
            "modified_image_path": modified_path,
            "operation": f"Median Filter Denoising (Kernel: {kernel_size}x{kernel_size})"
        })
    
    # Sharpen Image
    async def sharpen_image(self, request: Request, file: UploadFile = File(...)):
        image_data = await file.read()
        np_array = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
        
        if img is None:
            return HTMLResponse("Gagal membaca gambar", status_code=400)
        
        # Apply sharpening kernel
        kernel = np.array([[0, -1, 0], 
                          [-1, 5, -1], 
                          [0, -1, 0]])
        sharpened_img = cv2.filter2D(img, -1, kernel)
        
        original_path = self.save_image(img, "modul4_original")
        modified_path = self.save_image(sharpened_img, "modul4_sharpened")
        
        return self.templates.TemplateResponse("modules/modul4/result.html", {
            "request": request,
            "original_image_path": original_path,
            "modified_image_path": modified_path,
            "operation": "Image Sharpening"
        })
    
    # Complete Processing Pipeline
    async def complete_processing(self, request: Request, file: UploadFile = File(...)):
        image_data = await file.read()
        np_array = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
        
        if img is None:
            return HTMLResponse("Gagal membaca gambar", status_code=400)
        
        # Processing pipeline: Add noise -> Remove noise -> Sharpen
        noisy_img = self._add_salt_pepper_noise(img, 0.02, 0.02)
        denoised_img = cv2.medianBlur(noisy_img, 5)
        
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        final_img = cv2.filter2D(denoised_img, -1, kernel)
        
        # Save all steps
        original_path = self.save_image(img, "modul4_original")
        noisy_path = self.save_image(noisy_img, "modul4_step1_noisy")
        denoised_path = self.save_image(denoised_img, "modul4_step2_denoised")
        final_path = self.save_image(final_img, "modul4_step3_final")
        
        return self.templates.TemplateResponse("modules/modul4/pipeline.html", {
            "request": request,
            "original_path": original_path,
            "noisy_path": noisy_path,
            "denoised_path": denoised_path,
            "final_path": final_path
        })
    
    def _add_salt_pepper_noise(self, image, salt_prob=0.02, pepper_prob=0.02):
        noisy_image = np.copy(image)
        total_pixels = image.size
        
        num_salt = int(total_pixels * salt_prob)
        num_pepper = int(total_pixels * pepper_prob)
        
        # Add salt noise (white)
        for _ in range(num_salt):
            x = random.randint(0, image.shape[1] - 1)
            y = random.randint(0, image.shape[0] - 1)
            if len(image.shape) == 3:  # Color image
                noisy_image[y, x] = [255, 255, 255]
            else:  # Grayscale
                noisy_image[y, x] = 255
        
        # Add pepper noise (black)
        for _ in range(num_pepper):
            x = random.randint(0, image.shape[1] - 1)
            y = random.randint(0, image.shape[0] - 1)
            if len(image.shape) == 3:  # Color image
                noisy_image[y, x] = [0, 0, 0]
            else:  # Grayscale
                noisy_image[y, x] = 0
        
        return noisy_image
    
    def save_image(self, image, prefix):
        filename = f"{prefix}_{uuid4()}.png"
        path = os.path.join("static/uploads", filename)
        cv2.imwrite(path, image)
        return f"/static/uploads/{filename}"