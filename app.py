from fastapi import FastAPI, Request, File, UploadFile, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import os

# Import modules
from modules.modul1 import Modul1
from modules.modul2 import Modul2
from modules.modul3 import Modul3  # 🔥 NEW

app = FastAPI(title="PCD Praktikum 2024 - RINDI INDRIANI - 231511030")

# Static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Ensure directories exist
if not os.path.exists("static/uploads"):
    os.makedirs("static/uploads")
if not os.path.exists("static/histograms"):
    os.makedirs("static/histograms")

# Initialize modules
modul1 = Modul1(templates)
modul2 = Modul2(templates)
modul3 = Modul3(templates)  # 🔥 NEW

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

# 🔥 MODUL 3 ROUTES - NEW
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)