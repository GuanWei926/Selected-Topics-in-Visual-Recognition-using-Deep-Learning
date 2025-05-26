import os
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from tqdm import tqdm
import matplotlib.pyplot as plt

from train import PromptIRModel
import torchvision.transforms.functional as TF


def tta_predict(model, image_tensor):
    """
    image_tensor: (3, H, W)
    return: (3, H, W)
    """
    model.eval()
    results = []

    # 8種 augmentation
    transforms = [
        lambda x: x,                                       # 原圖
        TF.hflip,                                          # 左右
        TF.vflip,                                          # 上下
        lambda x: TF.hflip(TF.vflip(x)),                   # 左右+上下
        lambda x: TF.rotate(x, 90),                        # 旋轉90
        lambda x: TF.hflip(TF.rotate(x, 90)),              # 旋轉90 + 左右
        lambda x: TF.rotate(x, 270),                       # 旋轉270
        lambda x: TF.hflip(TF.rotate(x, 270)),             # 旋轉270 + 左右
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

    for f, f_inv in zip(transforms, inverse_transforms):
        img_aug = f(image_tensor)
        with torch.no_grad():
            out = model(img_aug).squeeze(0)
        out = f_inv(out)
        results.append(out)

    return torch.stack(results).mean(dim=0)


# load model
model = PromptIRModel.load_from_checkpoint(
    "../checkpoint/3loss_patch=160_wd=1e-4_bs=4_epoch=336_30.447.ckpt")
model.eval()
model.cuda()

# read test images
test_dir = "data/Test/degraded"
image = ["3.png", "91.png"]
transform = transforms.Compose([transforms.ToTensor()])

for filename in tqdm(sorted(os.listdir(test_dir))):
    if filename not in image:
        continue
    img_path = os.path.join(test_dir, filename)
    img = Image.open(img_path).convert("RGB")
    tensor_img = transform(img).unsqueeze(0).cuda()  # (1, 3, H, W)

    with torch.no_grad():
        restored = tta_predict(model, tensor_img)  # (1, 3, H, W)
        # restored = model(tensor_img)
        restored = restored.squeeze(0).cpu().clamp(0, 1)

    degraded_np = np.array(img)
    restored_np = (
        (restored.numpy().transpose(1, 2, 0) * 255)
        .round().astype(np.uint8))

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(degraded_np)
    axes[0].set_title("Degraded")
    axes[0].axis("off")

    axes[1].imshow(restored_np)
    axes[1].set_title("Restored")
    axes[1].axis("off")

    plt.suptitle(f"Comparison: {filename}", fontsize=14)
    plt.tight_layout()
    save_dir = "comparison_images"
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f"compare_{filename}"))
    plt.close()  # Close the figure to save memory
