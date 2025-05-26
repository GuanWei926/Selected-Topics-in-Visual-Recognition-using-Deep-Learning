import os
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from tqdm import tqdm

from train import PromptIRModel
import torchvision.transforms.functional as TF

# ----------------- Checkpoint Ensemble -----------------
ckpt_paths = [
    "../checkpoint/3loss_patch=160_wd=1e-4_bs=4_epoch=336_30.447.ckpt",
    "../checkpoint/3loss_patch=224_epoch=265_30.256.ckpt",
]

models = []
for path in ckpt_paths:
    model = PromptIRModel.load_from_checkpoint(path)
    model.eval()
    model.cuda()
    models.append(model)


# ----------------- TTA Predict  -----------------
def tta_predict(model, image_tensor):
    model.eval()
    results = []

    transforms_list = [
        lambda x: x,
        TF.hflip,
        TF.vflip,
        lambda x: TF.hflip(TF.vflip(x)),
        lambda x: TF.rotate(x, 90),
        lambda x: TF.hflip(TF.rotate(x, 90)),
        lambda x: TF.rotate(x, 270),
        lambda x: TF.hflip(TF.rotate(x, 270)),
    ]

    inverse_transforms = [
        lambda x: x,
        TF.hflip,
        TF.vflip,
        lambda x: TF.vflip(TF.hflip(x)),
        lambda x: TF.rotate(x, -90),
        lambda x: TF.rotate(TF.hflip(x), -90),
        lambda x: TF.rotate(x, -270),
        lambda x: TF.rotate(TF.hflip(x), -270),
    ]

    for f, f_inv in zip(transforms_list, inverse_transforms):
        img_aug = f(image_tensor.squeeze(0))  # remove batch dim for TTA
        with torch.no_grad():
            out = model(img_aug.unsqueeze(0).cuda()).squeeze(0)  # add batch
        out = f_inv(out)
        results.append(out)

    return torch.stack(results).mean(dim=0)  # (3, H, W)


# ----------------- 主推論流程 -----------------
test_dir = "data/Test/degraded"
transform = transforms.Compose([transforms.ToTensor()])
output_dict = {}

for filename in tqdm(sorted(os.listdir(test_dir))):
    if not filename.endswith((".png", ".jpg", ".jpeg")):
        continue

    img_path = os.path.join(test_dir, filename)
    img = Image.open(img_path).convert("RGB")
    tensor_img = transform(img).cuda()  # (3, H, W)

    model_outputs = []
    for model in models:
        output = tta_predict(model, tensor_img)  # (3, H, W)
        model_outputs.append(output)

    restored = torch.stack(model_outputs).mean(dim=0)  # ensemble 平均
    restored = restored.clamp(0, 1).cpu().numpy()
    restored = (restored * 255).round().astype(np.uint8)

    output_dict[filename] = restored

# ----------------- 儲存結果 -----------------
np.savez("pred_ensemble.npz", **output_dict)
