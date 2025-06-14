import os
from uuid import uuid4
from fastapi import File, UploadFile, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from skimage.exposure import match_histograms
import numpy as np
import cv2
import matplotlib.pyplot as plt

class Modul2:
    def __init__(self, templates):
        self.templates = templates
        self._ensure_directories()
    
    def _ensure_directories(self):
        if not os.path.exists("static/uploads"):
            os.makedirs("static/uploads")
        if not os.path.exists("static/histograms"):
            os.makedirs("static/histograms")
    
    async def home(self, request: Request):
        return self.templates.TemplateResponse("modules/modul2/home.html", {"request": request})
    
    async def upload_image(self, request: Request, file: UploadFile = File(...)):
        image_data = await file.read()
        np_array = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
        file_path = self.save_image(img, "modul2_uploaded")
        
        return self.templates.TemplateResponse("modules/modul2/result.html", {
            "request": request,
            "original_image_path": file_path,
            "modified_image_path": file_path
        })
    
    async def perform_operation(self, request: Request, file: UploadFile = File(...), 
                              operation: str = Form(...), value: int = Form(...)):
        image_data = await file.read()
        np_array = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
        
        original_path = self.save_image(img, "modul2_original")
        
        if operation == "add":
            result_img = cv2.add(img, np.full(img.shape, value, dtype=np.uint8))
        elif operation == "subtract":
            result_img = cv2.subtract(img, np.full(img.shape, value, dtype=np.uint8))
        elif operation == "max":
            result_img = np.maximum(img, np.full(img.shape, value, dtype=np.uint8))
        elif operation == "min":
            result_img = np.minimum(img, np.full(img.shape, value, dtype=np.uint8))
        elif operation == "inverse":
            result_img = cv2.bitwise_not(img)
        
        modified_path = self.save_image(result_img, "modul2_modified")
        
        return self.templates.TemplateResponse("modules/modul2/result.html", {
            "request": request,
            "original_image_path": original_path,
            "modified_image_path": modified_path
        })
    
    async def perform_logic_operation(self, request: Request, file1: UploadFile = File(...), 
                                    file2: UploadFile = File(None), operation: str = Form(...)):
        image_data1 = await file1.read()
        np_array1 = np.frombuffer(image_data1, np.uint8)
        img1 = cv2.imdecode(np_array1, cv2.IMREAD_COLOR)
        
        original_path = self.save_image(img1, "modul2_original")
        
        if operation == "not":
            result_img = cv2.bitwise_not(img1)
        else:
            if file2 is None:
                return HTMLResponse("Operasi AND dan XOR memerlukan dua gambar.", status_code=400)
            image_data2 = await file2.read()
            np_array2 = np.frombuffer(image_data2, np.uint8)
            img2 = cv2.imdecode(np_array2, cv2.IMREAD_COLOR)
            
            if operation == "and":
                result_img = cv2.bitwise_and(img1, img2)
            elif operation == "xor":
                result_img = cv2.bitwise_xor(img1, img2)
        
        modified_path = self.save_image(result_img, "modul2_modified")
        
        return self.templates.TemplateResponse("modules/modul2/result.html", {
            "request": request,
            "original_image_path": original_path,
            "modified_image_path": modified_path
        })
    
    async def convert_grayscale(self, request: Request, file: UploadFile = File(...)):
        image_data = await file.read()
        np_array = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
        
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        original_path = self.save_image(img, "modul2_original")
        modified_path = self.save_image(gray_img, "modul2_grayscale")
        
        return self.templates.TemplateResponse("modules/modul2/result.html", {
            "request": request,
            "original_image_path": original_path,
            "modified_image_path": modified_path
        })
    
    async def generate_histogram(self, request: Request, file: UploadFile = File(...)):
        image_data = await file.read()
        np_array = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
        
        if img is None:
            return HTMLResponse("Tidak dapat membaca gambar yang diunggah", status_code=400)
        
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        grayscale_histogram_path = self.save_histogram(gray_img, "modul2_grayscale")
        color_histogram_path = self.save_color_histogram(img)
        
        return self.templates.TemplateResponse("modules/modul2/histogram.html", {
            "request": request,
            "grayscale_histogram_path": grayscale_histogram_path,
            "color_histogram_path": color_histogram_path
        })
    
    async def equalize_histogram(self, request: Request, file: UploadFile = File(...)):
        image_data = await file.read()
        np_array = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(np_array, cv2.IMREAD_GRAYSCALE)
        
        equalized_img = cv2.equalizeHist(img)
        
        original_path = self.save_image(img, "modul2_original")
        modified_path = self.save_image(equalized_img, "modul2_equalized")
        
        return self.templates.TemplateResponse("modules/modul2/result.html", {
            "request": request,
            "original_image_path": original_path,
            "modified_image_path": modified_path
        })
    
    async def specify_histogram(self, request: Request, file: UploadFile = File(...), 
                              ref_file: UploadFile = File(...)):
        image_data = await file.read()
        ref_image_data = await ref_file.read()
        
        np_array = np.frombuffer(image_data, np.uint8)
        ref_np_array = np.frombuffer(ref_image_data, np.uint8)
        
        img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
        ref_img = cv2.imdecode(ref_np_array, cv2.IMREAD_COLOR)
        
        if img is None or ref_img is None:
            return HTMLResponse("Gambar utama atau gambar referensi tidak dapat dibaca.", status_code=400)
        
        specified_img = match_histograms(img, ref_img, channel_axis=-1)
        specified_img = np.clip(specified_img, 0, 255).astype('uint8')
        
        original_path = self.save_image(img, "modul2_original")
        modified_path = self.save_image(specified_img, "modul2_specified")
        
        return self.templates.TemplateResponse("modules/modul2/result.html", {
            "request": request,
            "original_image_path": original_path,
            "modified_image_path": modified_path
        })
    
    async def calculate_statistics(self, request: Request, file: UploadFile = File(...)):
        image_data = await file.read()
        np_array = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(np_array, cv2.IMREAD_GRAYSCALE)
        
        mean_intensity = np.mean(img)
        std_deviation = np.std(img)
        
        image_path = self.save_image(img, "modul2_statistics")
        
        return self.templates.TemplateResponse("modules/modul2/statistics.html", {
            "request": request,
            "mean_intensity": mean_intensity,
            "std_deviation": std_deviation,
            "image_path": image_path
        })
    
    def save_image(self, image, prefix):
        filename = f"{prefix}_{uuid4()}.png"
        path = os.path.join("static/uploads", filename)
        cv2.imwrite(path, image)
        return f"/static/uploads/{filename}"
    
    def save_histogram(self, image, prefix):
        histogram_path = f"static/histograms/{prefix}_{uuid4()}.png"
        plt.figure()
        plt.hist(image.ravel(), 256, [0, 256])
        plt.savefig(histogram_path)
        plt.close()
        return f"/{histogram_path}"
    
    def save_color_histogram(self, image):
        color_histogram_path = f"static/histograms/modul2_color_{uuid4()}.png"
        plt.figure()
        for i, color in enumerate(['b', 'g', 'r']):
            hist = cv2.calcHist([image], [i], None, [256], [0, 256])
            plt.plot(hist, color=color)
        plt.savefig(color_histogram_path)
        plt.close()
        return f"/{color_histogram_path}"