import cv2
import torch
import numpy as np
import matplotlib.pyplot as plot
from PIL import Image


#model_type = "DPT_Large"     # MiDaS v3 - Large     (highest accuracy, slowest inference speed)
#model_type = "DPT_Hybrid"   # MiDaS v3 - Hybrid    (medium accuracy, medium inference speed)
#model_type = "MiDaS_small"  # MiDaS v2.1 - Small   (lowest accuracy, highest inference speed)

def Midas_cv2():

    cap = cv2.VideoCapture(0)
    model = torch.hub.load('intel-isl/MiDaS', 'MiDaS_small')
    device = torch.device('cpu')
    model.to(device)
    model.eval()
    var = torch.hub.load('intel-isl/MiDaS', 'transforms')
    transform = var.small_transform 

    plot.ion()
    while(True):
        _, img = cap.read()
        img = cv2.resize(img, (600, 500)) 
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        input_img = transform(img).to(device)
        with torch.no_grad():
            prediction = model(input_img)
            prediction = torch.nn.functional.interpolate(prediction.unsqueeze(1),size=img.shape[:2],mode="bicubic",align_corners=False).squeeze()
        output = prediction.cpu().numpy()
        to_int = output.astype(int)
        oned = to_int.flatten()
        gray = cv2.normalize(to_int, None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        # img_uint = np.uint8(gray)
        heatmap = cv2.applyColorMap(gray, cv2.COLORMAP_INFERNO)
        
        distance = oned.item(1)
        print(distance)

        mean = np.mean(to_int)

        # depth_min = gray.min()
        # depth_max = gray.max() 
        # print(mean)
        # print(np.sort(mean)) 

        cv2.imshow('heat', heatmap)
        cv2.imshow('gray', gray)

        print('out:\n', oned)
        # print('int:\n', mean)
        print('gray\n', gray)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cv2.destroyAllWindows()


if __name__ == '__main__':
  Midas_cv2()