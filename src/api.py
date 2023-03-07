"""SegFormer api file"""
import os
import io
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi import HTTPException
from starlette.responses import StreamingResponse

from PIL import Image
# from acl.api import get_timestamp

from mmseg.apis import init_segmentor, inference_segmentor, show_result_solid, show_result_semi, show_class_names
from mmseg.core.evaluation import get_palette

import config 

segformer = FastAPI(
    title = config.Settings().title,
    description = config.Settings().description,
    version = config.Settings().version,
    openapi_tags = config.Settings().openapi_tags
)

segformer.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# create input dir if not alr available
INPUT_DIR = "./input"
if not os.path.isdir(INPUT_DIR):
    os.mkdir(INPUT_DIR)

# create output dir if not alr available
OUTPUT_DIR = "./output"
if not os.path.isdir(OUTPUT_DIR):
    os.mkdir(OUTPUT_DIR)

# setup config and checkpoint file (ade b5)
config_file = 'local_configs/segformer/B5/segformer.b5.640x640.ade.160k.py'
checkpoint_file = './checkpoints/segformer.b5.640x640.ade.160k.pth'

# Use load_from_local loader
model = init_segmentor(config_file, checkpoint_file, device='cuda:0')

@segformer.post("/solid", tags=['segformer'])
def create_solid_mask(image_binary: UploadFile = File(...)):
    '''Create SegFormer Solid Mask'''
    # timestamp = get_timestamp()

    try:
        image = Image.open(io.BytesIO(image_binary.file.read()))
        image = image.convert("RGB")
        image.save(os.path.join(INPUT_DIR, f"input_.jpg"), quality=100, subsampling=0)
    except IOError:
        raise HTTPException(status_code=422, detail="Invalid source image")

    try:
        filename = os.path.join(INPUT_DIR, f"input_.jpg")
        result = inference_segmentor(model, filename)
        result_tensor = show_result_solid(model, filename, result, get_palette('ade'))
        im = Image.fromarray(result_tensor)
        im.save(os.path.join(OUTPUT_DIR, f"output_.png"), quality=100, subsampling=0)
    except:
        print("CUDA out of memory")
        raise HTTPException(status_code=500, detail="CUDA out of memory")
    
    with open(os.path.join(OUTPUT_DIR, f"output_.png"), "rb") as o:
        res = o.read()
        o.close()

    return StreamingResponse(io.BytesIO(res), media_type="image/png")
    
@segformer.post("/semi", tags=['segformer'])
def create_semi_mask(image_binary: UploadFile = File(...)):
    '''Create SegFormer Semi Mask'''
    # timestamp = get_timestamp()

    try:
        image = Image.open(io.BytesIO(image_binary.file.read()))
        image = image.convert("RGB")
        image.save(os.path.join(INPUT_DIR, f"input_.jpg"), quality=100, subsampling=0)
    except IOError:
        raise HTTPException(status_code=422, detail="Invalid source image")

    try:
        filename = os.path.join(INPUT_DIR, f"input_.jpg")
        result = inference_segmentor(model, filename)
        result_tensor = show_result_semi(model, filename, result, get_palette('ade'))
        im = Image.fromarray(result_tensor)
        im.save(os.path.join(OUTPUT_DIR, f"output_.png"), quality=100, subsampling=0)
    except:
        print("CUDA out of memory")
        raise HTTPException(status_code=500, detail="CUDA out of memory")
    
    with open(os.path.join(OUTPUT_DIR, f"output_.png"), "rb") as o:
        res = o.read()
        o.close()

    return StreamingResponse(io.BytesIO(res), media_type="image/png")

@segformer.post("/get_class_names", tags=['segformer'])
def get_class_names(image_binary: UploadFile = File(...)):
    '''Get class names for uploaded image'''
    # timestamp = get_timestamp()

    try:
        image = Image.open(io.BytesIO(image_binary.file.read()))
        image = image.convert("RGB")
        image.save(os.path.join(INPUT_DIR, f"input_.jpg"), quality=100, subsampling=0)
    except IOError:
        raise HTTPException(status_code=422, detail="Invalid source image")

    try:
        filename = os.path.join(INPUT_DIR, f"input_.jpg")
        result = inference_segmentor(model, filename)
        class_names = show_class_names(model, filename, result, get_palette('ade'))   
    except:
        print("CUDA out of memory")
        raise HTTPException(status_code=500, detail="CUDA out of memory")

    return class_names
