import time
import matplotlib.pyplot as plt
from PIL import Image

import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.transforms.functional import to_pil_image
from torchcam.utils import overlay_mask

from recipro_cam_hook import *
from recipro_cam import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Test with: {device}")

model = models.resnet50(pretrained=True).to(device)
backbone = torch.nn.Sequential(*list(model.children())[:-2]).to(device)
cl_head = torch.nn.Sequential(list(model.children())[-2], 
                                torch.nn.Flatten(), 
                                list(model.children())[-1]).to(device)

input_file_name = '/local path/ILSVRC2012_cls/val/n06794110/ILSVRC2012_val_00011160.JPEG'
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
img = Image.open(input_file_name)
img = img.resize((224, 224))
input_tensor = transform(img).to(device)
img = np.asarray(img)

def recipro_cam_test():

    #torch_cam like hook method
    begin_time = time.time()
    recipro_cam = ReciproCamHook(model, device=device)
    cam, class_id = recipro_cam(input_tensor.unsqueeze(0))
    last_time = time.time()
    print('1. Recipro CAM Execution Time: ', last_time - begin_time, ' : ', class_id)

    result = overlay_mask(to_pil_image(img), to_pil_image(cam.detach(), mode='F'), alpha=0.5)
    plt.imshow(result); plt.axis('off'); plt.tight_layout(); plt.show()
    filename = './cam_results/test1.png'
    result.save(filename)

    #network separation method
    begin_time = time.time()
    recipro_cam = ReciproCam(backbone, cl_head, device=device)
    cam, class_id = recipro_cam(input_tensor.unsqueeze(0))
    last_time = time.time()
    print('2. Recipro CAM Execution Time: ', last_time - begin_time, ' : ', class_id)

    result = overlay_mask(to_pil_image(img), to_pil_image(cam.detach(), mode='F'), alpha=0.5)
    plt.imshow(result); plt.axis('off'); plt.tight_layout(); plt.show()
    filename = './cam_results/test2.png'
    result.save(filename)

    return 
    

if __name__ == '__main__':

    recipro_cam_test()

