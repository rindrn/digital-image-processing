import os
from uuid import uuid4
from fastapi import File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import numpy as np
import cv2

class Modul1:
    def __init__(self, templates):
        self.templates = templates
        
    async def home(self, request: Request):
        return self.templates.TemplateResponse("modules/modul1/home.html", {"request": request})
    
    async def upload_image(self, request: Request, file: UploadFile = File(...)):
        image_data = await file.read()
        file_extension = file.filename.split(".")[-1]
        filename = f"modul1_{uuid4()}.{file_extension}"
        file_path = os.path.join("static", "uploads", filename)

        with open(file_path, "wb") as f:
            f.write(image_data)

        np_array = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
        b, g, r = cv2.split(img)

        rgb_array = {"R": r.tolist(), "G": g.tolist(), "B": b.tolist()}

        return self.templates.TemplateResponse("modules/modul1/display.html", {
            "request": request,
            "image_path": f"/static/uploads/{filename}",
            "rgb_array": rgb_array
        })