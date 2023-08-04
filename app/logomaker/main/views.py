from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
from .apps import MainConfig
import base64
import numpy as np
import io
import torch
from PIL import Image
import json
from .pipeline import *

TOP_K = 3
np.set_printoptions(suppress=True)

# Create your views here.
def process_image(imageFile):
    imageFile = base64.b64decode(imageFile.split(",")[1])
    img = Image.open(io.BytesIO(imageFile))
    img = np.array(img)
    # 채널 하나로 만들기
    img = img[:,:,1]
    img = img.reshape(1, 28, 28, 1)
    image = torch.from_numpy(img).float()
    # 255로 나누기
    image = image / 255
    # 반전시키기
    image = 1 - image
    image = image.view(-1, 1, 28, 28)
    return image

def home(request):
    return render(request, "main/home.html")

def generate(request):
    if request.method == "POST":
        # Get text data from post request
        jsonObject = json.loads(request.body)
        text = jsonObject.get("text")

        # Get image data from text with VQGAN-CLIP
        img = inference(
            text = text, 
            seed = 2,
            step_size = 0.12,
            max_iterations = 200,
            width = 512,
            height = 512,
            init_image = '',
            init_weight = 0.004,
            target_images = '', 
            cutn = 64,
            cut_pow = 0.3,
            video_file = "test1"
        )
        npImage = display_result(img)
        # 
        # numpy 배열을 이미지로 변환
        npImage = Image.fromarray(npImage.astype(np.uint8))
        # 이미지를 base64로 인코딩
        buffer = io.BytesIO()
        npImage.save(buffer, format="JPEG")
        new_image_string = base64.b64encode(buffer.getvalue()).decode("utf-8")
        out_string = "data:image/jpeg;base64," + new_image_string

        return JsonResponse({"image": out_string})
    else:
        return render(request, "main/home.html")
