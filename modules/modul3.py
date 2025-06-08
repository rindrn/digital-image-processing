import os
from uuid import uuid4
from fastapi import File, UploadFile, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import numpy as np
import cv2
from io import BytesIO

class Modul3:
    def __init__(self, templates):
        self.templates = templates
        self._ensure_directories()
    
    def _ensure_directories(self):
        if not os.path.exists("static/uploads"):
            os.makedirs("static/uploads")
    
    async def home(self, request: Request):
        return self.templates.TemplateResponse("modules/modul3/home.html", {"request": request})
    
    # Utility functions
    def read_image_from_bytes(self, image_data):
        """Membaca gambar dari bytes"""
        np_array = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
        return img
    
    def save_image(self, image, prefix):
        """Simpan gambar ke static/uploads"""
        filename = f"modul3_{prefix}_{uuid4()}.png"
        path = os.path.join("static/uploads", filename)
        cv2.imwrite(path, image)
        return f"/static/uploads/{filename}"
    
    # Zero Padding
    async def apply_zero_padding(self, request: Request, file: UploadFile = File(...), 
                               padding_size: int = Form(10)):
        image_data = await file.read()
        img = self.read_image_from_bytes(image_data)
        
        if img is None:
            return HTMLResponse("Gagal membaca gambar", status_code=400)
        
        # Apply zero padding
        padded_img = cv2.copyMakeBorder(
            img, padding_size, padding_size, padding_size, padding_size, 
            cv2.BORDER_CONSTANT, value=[0, 0, 0]
        )
        
        original_path = self.save_image(img, "original")
        result_path = self.save_image(padded_img, "zero_padded")
        
        return self.templates.TemplateResponse("modules/modul3/result.html", {
            "request": request,
            "original_image_path": original_path,
            "result_image_path": result_path,
            "operation": f"Zero Padding (Size: {padding_size})",
            "description": f"Menambahkan border hitam dengan ukuran {padding_size} pixel di sekeliling gambar."
        })
    
    # Convolution
    async def apply_convolution(self, request: Request, file: UploadFile = File(...), 
                              kernel_type: str = Form("average")):
        image_data = await file.read()
        img = self.read_image_from_bytes(image_data)
        
        if img is None:
            return HTMLResponse("Gagal membaca gambar", status_code=400)
        
        # Define kernels
        kernels = {
            "average": (np.ones((3, 3), np.float32) / 9, "Average Filter - Menghaluskan gambar"),
            "sharpen": (np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]), "Sharpen Filter - Mempertajam detail"),
            "edge": (np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]), "Edge Detection - Mendeteksi tepi")
        }
        
        if kernel_type not in kernels:
            return HTMLResponse("Tipe kernel tidak valid", status_code=400)
        
        kernel, description = kernels[kernel_type]
        result_img = cv2.filter2D(img, -1, kernel)
        
        original_path = self.save_image(img, "original")
        result_path = self.save_image(result_img, f"convolution_{kernel_type}")
        
        return self.templates.TemplateResponse("modules/modul3/result.html", {
            "request": request,
            "original_image_path": original_path,
            "result_image_path": result_path,
            "operation": f"Convolution - {kernel_type.title()}",
            "description": description
        })
    
    # Frequency Filtering
    async def apply_filter(self, request: Request, file: UploadFile = File(...), 
                         filter_type: str = Form("low")):
        image_data = await file.read()
        img = self.read_image_from_bytes(image_data)
        
        if img is None:
            return HTMLResponse("Gagal membaca gambar", status_code=400)
        
        if filter_type == "low":
            # Low pass filter - Gaussian Blur
            result_img = cv2.GaussianBlur(img, (15, 15), 0)
            description = "Low Pass Filter - Menghilangkan noise dan detail halus"
        elif filter_type == "high":
            # High pass filter - Sharpen
            kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
            result_img = cv2.filter2D(img, -1, kernel)
            description = "High Pass Filter - Mempertahankan detail dan tepi"
        elif filter_type == "band":
            # Band pass filter
            low_pass = cv2.GaussianBlur(img, (9, 9), 0)
            high_pass = cv2.subtract(img, low_pass)
            result_img = cv2.add(low_pass, high_pass)
            description = "Band Pass Filter - Mempertahankan frekuensi tertentu"
        else:
            return HTMLResponse("Tipe filter tidak valid", status_code=400)
        
        original_path = self.save_image(img, "original")
        result_path = self.save_image(result_img, f"filter_{filter_type}")
        
        return self.templates.TemplateResponse("modules/modul3/result.html", {
            "request": request,
            "original_image_path": original_path,
            "result_image_path": result_path,
            "operation": f"{filter_type.title()} Pass Filter",
            "description": description
        })
    
    # Fourier Transform
    async def apply_fourier_transform(self, request: Request, file: UploadFile = File(...)):
        image_data = await file.read()
        img = self.read_image_from_bytes(image_data)
        
        if img is None:
            return HTMLResponse("Gagal membaca gambar", status_code=400)
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply Fourier Transform
        f = np.fft.fft2(gray)
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)  # +1 to avoid log(0)
        
        # Normalize to 0-255
        magnitude_spectrum = cv2.normalize(magnitude_spectrum, None, 0, 255, cv2.NORM_MINMAX)
        magnitude_spectrum = np.uint8(magnitude_spectrum)
        
        original_path = self.save_image(img, "original")
        result_path = self.save_image(magnitude_spectrum, "fourier_transform")
        
        return self.templates.TemplateResponse("modules/modul3/result.html", {
            "request": request,
            "original_image_path": original_path,
            "result_image_path": result_path,
            "operation": "Fourier Transform",
            "description": "Menampilkan spektrum frekuensi gambar dalam domain frekuensi"
        })
    
    # Noise Reduction
    async def reduce_periodic_noise(self, request: Request, file: UploadFile = File(...), 
                                  radius: int = Form(30)):
        image_data = await file.read()
        img = self.read_image_from_bytes(image_data)
        
        if img is None:
            return HTMLResponse("Gagal membaca gambar", status_code=400)
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply FFT
        f = np.fft.fft2(gray)
        fshift = np.fft.fftshift(f)
        
        # Create mask to remove periodic noise
        rows, cols = gray.shape
        crow, ccol = rows // 2, cols // 2
        mask = np.ones((rows, cols), np.uint8)
        mask[crow-radius:crow+radius, ccol-radius:ccol+radius] = 0
        
        # Apply mask and inverse FFT
        fshift = fshift * mask
        f_ishift = np.fft.ifftshift(fshift)
        img_back = np.fft.ifft2(f_ishift)
        img_back = np.abs(img_back)
        
        # Normalize
        img_back = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX)
        img_back = np.uint8(img_back)
        
        original_path = self.save_image(img, "original")
        result_path = self.save_image(img_back, "noise_reduced")
        
        return self.templates.TemplateResponse("modules/modul3/result.html", {
            "request": request,
            "original_image_path": original_path,
            "result_image_path": result_path,
            "operation": f"Noise Reduction (Radius: {radius})",
            "description": f"Mengurangi periodic noise menggunakan frequency domain filtering dengan radius {radius}"
        })