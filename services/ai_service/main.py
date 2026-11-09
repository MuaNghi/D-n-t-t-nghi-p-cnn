import json
from contextlib import asynccontextmanager
from io import BytesIO
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image, UnidentifiedImageError
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse


# ==========================================
# 1. CAU HINH DUONG DAN
# ==========================================

PROJECT_DIR = Path(__file__).resolve().parents[2]

# Model nhan dien benh lua
DISEASE_MODEL_PATH = (
    PROJECT_DIR / "models" / "rice_disease_model_finetuned_best.keras"
)
DISEASE_CLASS_NAMES_PATH = PROJECT_DIR / "models" / "class_names.json"

# Model nhan dien sau hai
PEST_MODEL_PATH = PROJECT_DIR / "models" / "pest_mobilenetv2_best.keras"
PEST_CLASS_NAMES_PATH = PROJECT_DIR / "models" / "pest_class_names.json"
PEST_CLASS_NAMES_FALLBACK_PATH = PROJECT_DIR / "reports" / "pest" / "class_names.json"

IMAGE_SIZE = (224, 224)
MAX_FILE_SIZE = 10 * 1024 * 1024


# ==========================================
# 2. DOC DANH SACH CAC LOP
# ==========================================

with open(DISEASE_CLASS_NAMES_PATH, "r", encoding="utf-8") as file:
    DISEASE_CLASS_NAMES = json.load(file)

PEST_CLASS_NAMES = []

try:
    if PEST_CLASS_NAMES_PATH.exists():
        with open(PEST_CLASS_NAMES_PATH, "r", encoding="utf-8") as file:
            PEST_CLASS_NAMES = json.load(file)
    elif PEST_CLASS_NAMES_FALLBACK_PATH.exists():
        with open(PEST_CLASS_NAMES_FALLBACK_PATH, "r", encoding="utf-8") as file:
            PEST_CLASS_NAMES = json.load(file)
except Exception as error:
    print(f"Khong doc duoc danh sach lop sau hai: {error}")


# ==========================================
# 3. TAI CAC MODEL KHI KHOI DONG SERVICE
# ==========================================

@asynccontextmanager
async def lifespan(app: FastAPI):

    if not DISEASE_MODEL_PATH.exists():
        raise RuntimeError(
            f"Khong tim thay model benh lua: {DISEASE_MODEL_PATH}"
        )

    print("Dang tai mo hinh nhan dien benh lua...")

    app.state.model = tf.keras.models.load_model(
        DISEASE_MODEL_PATH,
        compile=False,
    )

    print("Da tai mo hinh benh lua thanh cong!")
    print(f"So lop benh lua: {len(DISEASE_CLASS_NAMES)}")

    # Tai model sau hai doc lap
    app.state.pest_model = None

    try:
        if not PEST_MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Khong tim thay model sau hai: {PEST_MODEL_PATH}"
            )

        if not PEST_CLASS_NAMES:
            raise RuntimeError("Danh sach lop sau hai rong hoac khong hop le.")

        print("Dang tai mo hinh nhan dien sau hai...")

        app.state.pest_model = tf.keras.models.load_model(
            PEST_MODEL_PATH,
            compile=False,
        )

        print("Da tai mo hinh sau hai thanh cong!")
        print(f"So lop sau hai: {len(PEST_CLASS_NAMES)}")

    except Exception as error:
        app.state.pest_model = None
        print(f"CANH BAO: Khong tai duoc model sau hai: {error}")
        print("Chuc nang nhan dien benh lua van tiep tuc hoat dong.")

    yield

    app.state.model = None
    app.state.pest_model = None

    print("AI Service da dung.")


# ==========================================
# 4. KHOI TAO FASTAPI
# ==========================================

app = FastAPI(
    title="LuaAI - AI Recognition Service",
    description="Dich vu nhan dien benh lua va sau hai tren cay lua.",
    version="2.0.0",
    lifespan=lifespan,
)


# ==========================================
# 5. CAC HAM XU LY ANH VA DU DOAN
# ==========================================

async def predict_image(
    file: UploadFile,
    model,
    class_names,
    model_name: str,
):

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="Vui long chon anh de nhan dien.",
        )

    contents = await file.read()

    if not contents:
        raise HTTPException(
            status_code=400,
            detail="File anh rong.",
        )

    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail="Kich thuoc anh vuot qua gioi han 10 MB.",
        )

    try:
        image = Image.open(BytesIO(contents))
        image = image.convert("RGB")
        image = image.resize(IMAGE_SIZE)

    except (UnidentifiedImageError, OSError, ValueError):
        raise HTTPException(
            status_code=400,
            detail="File khong phai anh hop le.",
        )

    # Chuyen anh RGB thanh mang NumPy.
    # Khong tu chuan hoa ve [-1, 1] tai day:
    # Model MobileNetV2 da co Rescaling ben trong kien truc.
    image_array = np.asarray(
        image,
        dtype=np.float32,
    )

    image_array = np.expand_dims(image_array, axis=0)

    if model is None:
        raise HTTPException(
            status_code=503,
            detail=f"Mo hinh {model_name} chua san sang.",
        )

    try:
        predictions = model.predict(
            image_array,
            verbose=0,
        )[0]

    except Exception as error:
        print(f"Loi du doan {model_name}: {error}")

        raise HTTPException(
            status_code=500,
            detail=f"Khong the thuc hien du doan bang mo hinh {model_name}.",
        )

    if len(predictions) != len(class_names):
        raise HTTPException(
            status_code=500,
            detail="So luong ket qua khong khop voi danh sach lop.",
        )

    predicted_index = int(np.argmax(predictions))
    predicted_class = class_names[predicted_index]
    confidence = float(predictions[predicted_index])

    probabilities = {
        class_name: round(float(probability) * 100, 2)
        for class_name, probability in zip(
            class_names,
            predictions,
        )
    }

    return {
        "success": True,
        "filename": file.filename,
        "model_name": model_name,
        "predicted_class": predicted_class,
        "confidence": round(confidence * 100, 2),
        "unit": "percent",
        "probabilities": probabilities,
    }


# ==========================================
# 6. TRANG CHU
# ==========================================

@app.get("/")
async def home():
    return {
        "service": "LuaAI AI Recognition Service",
        "message": "Rice disease and pest recognition service",
        "status": "running",
    }


@app.get("/test")
async def test_page():
    html_path = Path(__file__).resolve().parent / "test_page.html"

    if not html_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Khong tim thay trang web kiem thu.",
        )

    return FileResponse(
        html_path,
        media_type="text/html",
    )


# ==========================================
# 7. KIEM TRA SUC KHOE
# ==========================================

@app.get("/health")
async def health_check():
    return {
        "service": "ai_recognition_service",
        "status": "healthy",
        "disease_model_loaded": getattr(app.state, "model", None) is not None,
        "pest_model_loaded": getattr(app.state, "pest_model", None) is not None,
    }


# ==========================================
# 8. TRANG THAI MODEL BENH LUA
# ==========================================

@app.get("/model/status")
async def model_status():
    return {
        "model_name": "rice_disease_model_finetuned_best",
        "model_path": str(DISEASE_MODEL_PATH),
        "model_exists": DISEASE_MODEL_PATH.exists(),
        "model_loaded": getattr(app.state, "model", None) is not None,
        "class_names_path": str(DISEASE_CLASS_NAMES_PATH),
        "class_names_exists": DISEASE_CLASS_NAMES_PATH.exists(),
        "classes": DISEASE_CLASS_NAMES,
        "number_of_classes": len(DISEASE_CLASS_NAMES),
    }


# ==========================================
# 9. TRANG THAI MODEL SAU HAI
# ==========================================

@app.get("/model/pest/status")
async def pest_model_status():
    return {
        "model_name": "pest_mobilenetv2_best",
        "model_path": str(PEST_MODEL_PATH),
        "model_exists": PEST_MODEL_PATH.exists(),
        "model_loaded": getattr(app.state, "pest_model", None) is not None,
        "class_names_path": str(PEST_CLASS_NAMES_PATH),
        "class_names_exists": PEST_CLASS_NAMES_PATH.exists(),
        "classes": PEST_CLASS_NAMES,
        "number_of_classes": len(PEST_CLASS_NAMES),
    }


# ==========================================
# 10. API NHAN DIEN BENH LA LUA
# ==========================================

@app.post("/predict")
async def predict_disease(file: UploadFile = File(...)):

    model = getattr(app.state, "model", None)

    return await predict_image(
        file=file,
        model=model,
        class_names=DISEASE_CLASS_NAMES,
        model_name="rice_disease_model_finetuned_best",
    )


# ==========================================
# 11. API NHAN DIEN SAU HAI
# ==========================================

@app.post("/predict/pest")
async def predict_pest(file: UploadFile = File(...)):

    model = getattr(app.state, "pest_model", None)

    return await predict_image(
        file=file,
        model=model,
        class_names=PEST_CLASS_NAMES,
        model_name="pest_mobilenetv2_best",
    )
