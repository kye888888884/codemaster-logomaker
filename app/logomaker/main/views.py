from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
from .apps import MainConfig
import base64
import numpy as np
import io
import torch
from PIL import Image

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

def mnist(request):
    if request.method == "POST":
        # Get formData
        imageFile = request.POST.get("image_data", "")
        # Process image
        img = process_image(imageFile)

        pred_num = MainConfig.deepmodel.predict(img)
        pred_num = pred_num.numpy()
        # 소수점 3자리까지 표시
        pred_num = np.round(pred_num, 4)
        # pred_num 배열에서 상위 K개의 인덱스를 가져온다.
        top_index = np.argsort(pred_num)[0][::-1][:TOP_K]
        # 상위 K개의 값
        top_value = pred_num[0][top_index]
        print(top_index, top_value)

        res = {
            str(x): {
                "index": str(top_index[x]),
                "value": str(top_value[x]),
            }
            for x in range(TOP_K)
        }

        return JsonResponse(res)
    else:
        return render(request, "main/home.html")
