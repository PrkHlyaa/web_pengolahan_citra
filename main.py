import os
from uuid import uuid4
from fastapi import FastAPI, File, UploadFile, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from skimage.exposure import match_histograms

import numpy as np
import cv2
import matplotlib.pyplot as plt

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

if not os.path.exists("static/uploads"):
    os.makedirs("static/uploads")

if not os.path.exists("static/histograms"):
    os.makedirs("static/histograms")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})

@app.post("/upload/", response_class=HTMLResponse)
async def upload_image(request: Request, file: UploadFile = File(...)):
    image_data = await file.read()
    np_array = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

    file_path = save_image(img, "uploaded")

    return templates.TemplateResponse("result.html", {
        "request": request,
        "original_image_path": file_path,
        "modified_image_path": file_path
    })

@app.post("/operation/", response_class=HTMLResponse)
async def perform_operation(
    request: Request,
    file: UploadFile = File(...),
    operation: str = Form(...),
    value: int = Form(...)
):
    image_data = await file.read()
    np_array = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

    original_path = save_image(img, "original")

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

    modified_path = save_image(result_img, "modified")

    return templates.TemplateResponse("result.html", {
        "request": request,
        "original_image_path": original_path,
        "modified_image_path": modified_path
    })

@app.post("/logic_operation/", response_class=HTMLResponse)
async def perform_logic_operation(
    request: Request,
    file1: UploadFile = File(...),
    file2: UploadFile = File(None),
    operation: str = Form(...)
):
    image_data1 = await file1.read()
    np_array1 = np.frombuffer(image_data1, np.uint8)
    img1 = cv2.imdecode(np_array1, cv2.IMREAD_COLOR)

    original_path = save_image(img1, "original")

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

    modified_path = save_image(result_img, "modified")

    return templates.TemplateResponse("result.html", {
        "request": request,
        "original_image_path": original_path,
        "modified_image_path": modified_path
    })
@app.get("/grayscale/", response_class=HTMLResponse)
async def grayscale_form(request: Request):
    # Menampilkan form untuk upload gambar ke grayscale
    return templates.TemplateResponse("grayscale.html", {"request": request})

@app.post("/grayscale/", response_class=HTMLResponse)
async def convert_grayscale(request: Request, file: UploadFile = File(...)):
    image_data = await file.read()
    np_array = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    original_path = save_image(img, "original")
    modified_path = save_image(gray_img, "grayscale")

    return templates.TemplateResponse("result.html", {
        "request": request,
        "original_image_path": original_path,
        "modified_image_path": modified_path
    })

@app.get("/histogram/", response_class=HTMLResponse)
async def histogram_form(request: Request):
    # Menampilkan halaman untuk upload gambar untuk histogram
    return templates.TemplateResponse("histogram.html", {"request": request})

@app.post("/histogram/", response_class=HTMLResponse)
async def generate_histogram(request: Request, file: UploadFile = File(...)):
    image_data = await file.read()
    np_array = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

    # Pastikan gambar berhasil diimpor
    if img is None:
        return HTMLResponse("Tidak dapat membaca gambar yang diunggah", status_code=400)

    # Buat histogram grayscale dan berwarna
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    grayscale_histogram_path = save_histogram(gray_img, "grayscale")

    color_histogram_path = save_color_histogram(img)

    return templates.TemplateResponse("histogram.html", {
        "request": request,
        "grayscale_histogram_path": grayscale_histogram_path,
        "color_histogram_path": color_histogram_path
    })



@app.get("/equalize/", response_class=HTMLResponse)
async def equalize_form(request: Request):
    # Menampilkan halaman untuk upload gambar untuk equalisasi histogram
    return templates.TemplateResponse("equalize.html", {"request": request})

@app.post("/equalize/", response_class=HTMLResponse)
async def equalize_histogram(request: Request, file: UploadFile = File(...)):
    image_data = await file.read()
    np_array = np.frombuffer(image_data, np.uint8)
    # img = cv2.imdecode(np_array, cv2.IMREAD_GRAYSCALE)
    img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

    # equalized_img = cv2.equalizeHist(img)

    # Konversi gambar ke ruang warna YUV (Y: Luminance, U/V: Chroma)
    yuv_img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)

    # Lakukan ekualisasi hanya pada saluran Y (intensitas)
    yuv_img[:,:,0] = cv2.equalizeHist(yuv_img[:,:,0])

    # Konversi kembali ke ruang warna BGR
    equalized_img = cv2.cvtColor(yuv_img, cv2.COLOR_YUV2BGR)

    original_path = save_image(img, "original")
    modified_path = save_image(equalized_img, "equalized")

    return templates.TemplateResponse("result.html", {
        "request": request,
        "original_image_path": original_path,
        "modified_image_path": modified_path
    })

@app.get("/specify/", response_class=HTMLResponse)
async def specify_form(request: Request):
    # Menampilkan halaman untuk upload gambar dan referensi untuk spesifikasi histogram
    return templates.TemplateResponse("specify.html", {"request": request})

@app.post("/specify/", response_class=HTMLResponse)
async def specify_histogram(request: Request, file: UploadFile = File(...), ref_file: UploadFile = File(...)):
    # Baca gambar yang diunggah dan gambar referensi
    image_data = await file.read()
    ref_image_data = await ref_file.read()

    np_array = np.frombuffer(image_data, np.uint8)
    ref_np_array = np.frombuffer(ref_image_data, np.uint8)
		
		#jika ingin grayscale
    #img = cv2.imdecode(np_array, cv2.IMREAD_GRAYSCALE)
    #ref_img = cv2.imdecode(ref_np_array, cv2.IMREAD_GRAYSCALE)

    img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)  # Membaca gambar dalam format BGR
    ref_img = cv2.imdecode(ref_np_array, cv2.IMREAD_COLOR)  # Membaca gambar referensi dalam format BGR


    if img is None or ref_img is None:
        return HTMLResponse("Gambar utama atau gambar referensi tidak dapat dibaca.", status_code=400)

    # Spesifikasi histogram menggunakan match_histograms dari skimage #grayscale
    #specified_img = match_histograms(img, ref_img, multichannel=False)
		    # Spesifikasi histogram menggunakan match_histograms dari skimage untuk gambar berwarna
    specified_img = match_histograms(img, ref_img, channel_axis=-1)
    # Konversi kembali ke format uint8 jika diperlukan
    specified_img = np.clip(specified_img, 0, 255).astype('uint8')

    original_path = save_image(img, "original")
    modified_path = save_image(specified_img, "specified")

    return templates.TemplateResponse("result.html", {
        "request": request,
        "original_image_path": original_path,
        "modified_image_path": modified_path
    })

@app.get("/statistics/", response_class=HTMLResponse)
async def statistics_form(request: Request):
    return templates.TemplateResponse("statistics.html", {"request": request})

@app.post("/statistics/", response_class=HTMLResponse)
async def calculate_statistics(request: Request, file: UploadFile = File(...)):
    image_data = await file.read()
    np_array = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(np_array, cv2.IMREAD_GRAYSCALE)

    mean_intensity = np.mean(img)
    std_deviation = np.std(img)

    image_path = save_image(img, "statistics")

    return templates.TemplateResponse("statistics.html", {
        "request": request,
        "mean_intensity": mean_intensity,
        "std_deviation": std_deviation,
        "image_path": image_path
    })

def save_image(image, prefix):
    filename = f"{prefix}_{uuid4()}.png"
    path = os.path.join("static/uploads", filename)
    cv2.imwrite(path, image)
    return f"/static/uploads/{filename}"

def save_histogram(image, prefix):
    histogram_path = f"static/histograms/{prefix}_{uuid4()}.png"
    plt.figure()
    plt.hist(image.ravel(), 256, [0, 256])
    plt.savefig(histogram_path)
    plt.close()
    return f"/{histogram_path}"

def save_color_histogram(image):
    color_histogram_path = f"static/histograms/color_{uuid4()}.png"
    plt.figure()
    for i, color in enumerate(['b', 'g', 'r']):
        hist = cv2.calcHist([image], [i], None, [256], [0, 256])
        plt.plot(hist, color=color)
    plt.savefig(color_histogram_path)
    plt.close()
    return f"/{color_histogram_path}"

# Tambahkan endpoint baru dari kode Colab
@app.get("/convolution/", response_class=HTMLResponse)
async def convolution_form(request: Request):
    return templates.TemplateResponse("convolution.html", {"request": request})

@app.post("/convolution/", response_class=HTMLResponse)
async def apply_convolution(
    request: Request,
    file: UploadFile = File(...),
    kernel_type: str = Form(...)
):
    image_data = await file.read()
    np_array = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

    if kernel_type == "average":
        kernel = np.ones((3, 3), np.float32) / 9
    elif kernel_type == "sharpen":
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    elif kernel_type == "edge":
        kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
    else:
        return HTMLResponse("Jenis kernel tidak valid.", status_code=400)

    output_img = cv2.filter2D(img, -1, kernel)

    original_path = save_image(img, "original")
    modified_path = save_image(output_img, f"{kernel_type}_convolution")

    return templates.TemplateResponse("result.html", {
        "request": request,
        "original_image_path": original_path,
        "modified_image_path": modified_path
    })


@app.get("/padding/", response_class=HTMLResponse)
async def padding_form(request: Request):
    return templates.TemplateResponse("padding.html", {"request": request})

@app.post("/padding/", response_class=HTMLResponse)
async def apply_zero_padding(
    request: Request,
    file: UploadFile = File(...),
    padding_size: int = Form(...)
):
    image_data = await file.read()
    np_array = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

    padded_img = cv2.copyMakeBorder(
        img,
        padding_size, padding_size, padding_size, padding_size,
        cv2.BORDER_CONSTANT,
        value=[0, 0, 0]
    )
    
    original_path = save_image(img, "original")
    modified_path = save_image(padded_img, "padded")

    return templates.TemplateResponse("result.html", {
        "request": request,
        "original_image_path": original_path,
        "modified_image_path": modified_path
    })


@app.get("/spatial_filter/", response_class=HTMLResponse)
async def spatial_filter_form(request: Request):
    return templates.TemplateResponse("spatial_filter.html", {"request": request})

@app.post("/spatial_filter/", response_class=HTMLResponse)
async def apply_spatial_filter(
    request: Request,
    file: UploadFile = File(...),
    filter_type: str = Form(...)
):
    image_data = await file.read()
    np_array = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

    if filter_type == "low":
        filtered_img = cv2.GaussianBlur(img, (5, 5), 0)
    elif filter_type == "high":
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        filtered_img = cv2.filter2D(img, -1, kernel)
    elif filter_type == "band":
        low_pass = cv2.GaussianBlur(img, (9, 9), 0)
        high_pass = img - low_pass
        filtered_img = low_pass + high_pass
    else:
        return HTMLResponse("Jenis filter tidak valid.", status_code=400)

    original_path = save_image(img, "original")
    modified_path = save_image(filtered_img, f"{filter_type}_filter")

    return templates.TemplateResponse("result.html", {
        "request": request,
        "original_image_path": original_path,
        "modified_image_path": modified_path
    })


@app.get("/fourier/", response_class=HTMLResponse)
async def fourier_transform_form(request: Request):
    return templates.TemplateResponse("fourier_transform.html", {"request": request})

@app.post("/fourier/", response_class=HTMLResponse)
async def apply_fourier_transform(request: Request, file: UploadFile = File(...)):
    image_data = await file.read()
    np_array = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift))
    
    # Simpan magnitude spectrum sebagai gambar
    # Normalisasi agar dapat disimpan sebagai gambar
    magnitude_spectrum = np.uint8(magnitude_spectrum / np.max(magnitude_spectrum) * 255)

    original_path = save_image(img, "original")
    modified_path = save_image(magnitude_spectrum, "fourier_transform")

    return templates.TemplateResponse("result.html", {
        "request": request,
        "original_image_path": original_path,
        "modified_image_path": modified_path
    })


@app.get("/noise_reduction/", response_class=HTMLResponse)
async def noise_reduction_form(request: Request):
    return templates.TemplateResponse("noise_reduction.html", {"request": request})

@app.post("/noise_reduction/", response_class=HTMLResponse)
async def reduce_periodic_noise(request: Request, file: UploadFile = File(...)):
    image_data = await file.read()
    np_array = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)

    rows, cols = gray.shape
    crow, ccol = rows // 2, cols // 2
    mask = np.ones((rows, cols), np.uint8)
    r = 30
    mask[crow-r:crow+r, ccol-r:ccol+r] = 0

    fshift = fshift * mask
    f_ishift = np.fft.ifftshift(fshift)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)

    # Konversi ke format uint8
    img_back = np.uint8(img_back)
    
    original_path = save_image(img, "original")
    modified_path = save_image(img_back, "noise_reduction")

    return templates.TemplateResponse("result.html", {
        "request": request,
        "original_image_path": original_path,
        "modified_image_path": modified_path
    })

def save_image(image, prefix):
    filename = f"{prefix}_{uuid4()}.png"
    path = os.path.join("static/uploads", filename)
    cv2.imwrite(path, image)
    return f"/static/uploads/{filename}"

def save_histogram(image, prefix):
    histogram_path = f"static/histograms/{prefix}_{uuid4()}.png"
    plt.figure()
    plt.hist(image.ravel(), 256, [0, 256])
    plt.savefig(histogram_path)
    plt.close()
    return f"/{histogram_path}"

def save_color_histogram(image):
    color_histogram_path = f"static/histograms/color_{uuid4()}.png"
    plt.figure()
    for i, color in enumerate(['b', 'g', 'r']):
        hist = cv2.calcHist([image], [i], None, [256], [0, 256])
        plt.plot(hist, color=color)
    plt.savefig(color_histogram_path)
    plt.close()
    return f"/{color_histogram_path}"