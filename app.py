from fastapi import FastAPI, Request, File, UploadFile, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import os

# Import modules
from modules.modul1 import Modul1
from modules.modul2 import Modul2
from modules.modul3 import Modul3
from modules.modul4 import Modul4
from modules.modul5 import Modul5  # 🔥 NEW

app = FastAPI(title="PCD Praktikum 2024 - RINDI INDRIANI - 231511030")

# Static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Ensure directories exist
directories = [
    "static/uploads",
    "static/histograms",
    "static/dataset",
    "static/processed_dataset"
]

for directory in directories:
    if not os.path.exists(directory):
        os.makedirs(directory)

# Initialize modules
modul1 = Modul1(templates)
modul2 = Modul2(templates)
modul3 = Modul3(templates)
modul4 = Modul4(templates)
modul5 = Modul5(templates)  # 🔥 NEW

# Main route
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# MODUL 1 ROUTES
@app.get("/modul1/", response_class=HTMLResponse)
async def modul1_home(request: Request):
    return await modul1.home(request)

@app.post("/modul1/upload/", response_class=HTMLResponse)
async def modul1_upload(request: Request, file: UploadFile = File(...)):
    return await modul1.upload_image(request, file)

# MODUL 2 ROUTES  
@app.get("/modul2/", response_class=HTMLResponse)
async def modul2_home(request: Request):
    return await modul2.home(request)

@app.post("/modul2/upload/", response_class=HTMLResponse)
async def modul2_upload(request: Request, file: UploadFile = File(...)):
    return await modul2.upload_image(request, file)

@app.post("/modul2/operation/", response_class=HTMLResponse)
async def modul2_operation(request: Request, file: UploadFile = File(...), 
                          operation: str = Form(...), value: int = Form(...)):
    return await modul2.perform_operation(request, file, operation, value)

@app.post("/modul2/logic_operation/", response_class=HTMLResponse)
async def modul2_logic_operation(request: Request, file1: UploadFile = File(...), 
                                file2: UploadFile = File(None), operation: str = Form(...)):
    return await modul2.perform_logic_operation(request, file1, file2, operation)

@app.get("/modul2/grayscale/", response_class=HTMLResponse)
async def modul2_grayscale_form(request: Request):
    return templates.TemplateResponse("modules/modul2/grayscale.html", {"request": request})

@app.post("/modul2/grayscale/", response_class=HTMLResponse)
async def modul2_grayscale(request: Request, file: UploadFile = File(...)):
    return await modul2.convert_grayscale(request, file)

@app.get("/modul2/histogram/", response_class=HTMLResponse)
async def modul2_histogram_form(request: Request):
    return templates.TemplateResponse("modules/modul2/histogram.html", {"request": request})

@app.post("/modul2/histogram/", response_class=HTMLResponse)
async def modul2_histogram(request: Request, file: UploadFile = File(...)):
    return await modul2.generate_histogram(request, file)

@app.get("/modul2/equalize/", response_class=HTMLResponse)
async def modul2_equalize_form(request: Request):
    return templates.TemplateResponse("modules/modul2/equalize.html", {"request": request})

@app.post("/modul2/equalize/", response_class=HTMLResponse)
async def modul2_equalize(request: Request, file: UploadFile = File(...)):
    return await modul2.equalize_histogram(request, file)

@app.get("/modul2/specify/", response_class=HTMLResponse)
async def modul2_specify_form(request: Request):
    return templates.TemplateResponse("modules/modul2/specify.html", {"request": request})

@app.post("/modul2/specify/", response_class=HTMLResponse)
async def modul2_specify(request: Request, file: UploadFile = File(...), ref_file: UploadFile = File(...)):
    return await modul2.specify_histogram(request, file, ref_file)

@app.post("/modul2/statistics/", response_class=HTMLResponse)
async def modul2_statistics(request: Request, file: UploadFile = File(...)):
    return await modul2.calculate_statistics(request, file)

# MODUL 3 ROUTES
@app.get("/modul3/", response_class=HTMLResponse)
async def modul3_home(request: Request):
    return await modul3.home(request)

@app.post("/modul3/zero_padding/", response_class=HTMLResponse)
async def modul3_zero_padding(request: Request, file: UploadFile = File(...), 
                             padding_size: int = Form(10)):
    return await modul3.apply_zero_padding(request, file, padding_size)

@app.post("/modul3/convolution/", response_class=HTMLResponse)
async def modul3_convolution(request: Request, file: UploadFile = File(...), 
                            kernel_type: str = Form("average")):
    return await modul3.apply_convolution(request, file, kernel_type)

@app.post("/modul3/filter/", response_class=HTMLResponse)
async def modul3_filter(request: Request, file: UploadFile = File(...), 
                       filter_type: str = Form("low")):
    return await modul3.apply_filter(request, file, filter_type)

@app.post("/modul3/fourier/", response_class=HTMLResponse)
async def modul3_fourier(request: Request, file: UploadFile = File(...)):
    return await modul3.apply_fourier_transform(request, file)

@app.post("/modul3/reduce_noise/", response_class=HTMLResponse)
async def modul3_reduce_noise(request: Request, file: UploadFile = File(...), 
                             radius: int = Form(30)):
    return await modul3.reduce_periodic_noise(request, file, radius)

# MODUL 4 ROUTES
@app.get("/modul4/", response_class=HTMLResponse)
async def modul4_home(request: Request):
    return await modul4.home(request)

@app.post("/modul4/detect_faces/", response_class=HTMLResponse)
async def modul4_detect_faces(request: Request, file: UploadFile = File(...)):
    return await modul4.detect_faces_upload(request, file)

@app.post("/modul4/add_to_dataset/", response_class=HTMLResponse)
async def modul4_add_to_dataset(request: Request, name: str = Form(...), 
                               file: UploadFile = File(...)):
    return await modul4.add_to_dataset(request, name, file)

@app.post("/modul4/add_noise/", response_class=HTMLResponse)
async def modul4_add_noise(request: Request, file: UploadFile = File(...), 
                          salt_prob: float = Form(0.02), pepper_prob: float = Form(0.02)):
    return await modul4.add_noise(request, file, salt_prob, pepper_prob)

@app.post("/modul4/remove_noise/", response_class=HTMLResponse)
async def modul4_remove_noise(request: Request, file: UploadFile = File(...), 
                             method: str = Form("median"), kernel_size: int = Form(3)):
    return await modul4.remove_noise(request, file, method, kernel_size)

@app.post("/modul4/sharpen/", response_class=HTMLResponse)
async def modul4_sharpen(request: Request, file: UploadFile = File(...),
                        method: str = Form("kernel")):
    return await modul4.sharpen(request, file, method)

@app.post("/modul4/advanced_convolution/", response_class=HTMLResponse)
async def modul4_advanced_convolution(request: Request, file: UploadFile = File(...), 
                                    operation: str = Form("blur")):
    return await modul4.advanced_convolution(request, file, operation)

# 🔥 MODUL 5 ROUTES - NEW
@app.get("/modul5/", response_class=HTMLResponse)
async def modul5_home(request: Request):
    return await modul5.home(request)

@app.post("/modul5/freeman_chain_code/", response_class=HTMLResponse)
async def modul5_freeman_chain_code(request: Request, file: UploadFile = File(...)):
    return await modul5.process_freeman_chain_code(request, file)

@app.post("/modul5/canny_edge_detection/", response_class=HTMLResponse)
async def modul5_canny_edge_detection(request: Request, file: UploadFile = File(...), 
                                     low_threshold: int = Form(50), 
                                     high_threshold: int = Form(150)):
    return await modul5.process_canny_edge_detection(request, file, low_threshold, high_threshold)

@app.post("/modul5/integral_projection/", response_class=HTMLResponse)
async def modul5_integral_projection(request: Request, file: UploadFile = File(...)):
    return await modul5.process_integral_projection(request, file)

@app.get("/modul5/demo/", response_class=HTMLResponse)
async def modul5_demo(request: Request):
    return await modul5.demo_analysis(request)

@app.post("/modul5/complete_analysis/", response_class=HTMLResponse)
async def modul5_complete_analysis(request: Request, file: UploadFile = File(...)):
    return await modul5.complete_analysis(request, file)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)