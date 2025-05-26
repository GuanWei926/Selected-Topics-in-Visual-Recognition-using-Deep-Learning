import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from utils.dataset_utils import PromptTrainDataset, PromptValDataset
from net.model import PromptIR
from utils.schedulers import LinearWarmupCosineAnnealingLR
from options import options as opt
from pytorch_msssim import SSIM
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger, TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from torchvision.models import vgg16
from torchvision.transforms import Normalize


checkpoint_callback = ModelCheckpoint(
    monitor="val_psnr",         # 監控 val_psnr
    mode="max",                 # 越大越好
    save_top_k=5,               # 只儲存最好的那個
    filename="{epoch:02d}-{val_psnr:.3f}",  # 儲存檔名格式
    dirpath=opt.ckpt_dir        # 儲存資料夾
)


class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, x, y):
        return torch.mean(torch.sqrt((x - y) ** 2 + self.eps ** 2))


class PromptIRModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.net = PromptIR(decoder=True)
        self.loss_fn = nn.L1Loss()
        self.l1 = nn.L1Loss()
        self.ssim = SSIM(data_range=1.0, size_average=True, channel=3)
        self.charbonnier = CharbonnierLoss()
        self.vgg = vgg16(pretrained=True).features[:16].eval().cuda()
        for param in self.vgg.parameters():
            param.requires_grad = False

        self.normalize = Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        ([clean_name, de_id], degrad_patch, clean_patch) = batch
        restored = self.net(degrad_patch)

        # loss = self.loss_fn(restored, clean_patch)
        loss = (
            self.l1(restored, clean_patch)
            + 0.2*(1 - self.ssim(restored, clean_patch))
            + 0.05*self.perceptual_loss(restored, clean_patch)
        )

        self.log(
            "train_loss", loss, on_step=False,
            on_epoch=True, prog_bar=True, sync_dist=True
        )
        return loss

    def validation_step(self, batch, batch_idx):
        ([clean_name, de_id], degrad_patch, clean_patch) = batch
        restored = self.net(degrad_patch)
        # loss = self.loss_fn(restored, clean_patch)
        loss = (
            self.l1(restored, clean_patch)
            + 0.2*(1 - self.ssim(restored, clean_patch))
            + 0.05*self.perceptual_loss(restored, clean_patch)
        )

        psnr = self._calculate_psnr(restored, clean_patch)
        self.log(
            "val_loss", loss, on_step=False,
            on_epoch=True, prog_bar=True, sync_dist=True)
        self.log(
            "val_psnr", psnr, on_step=False,
            on_epoch=True, prog_bar=True, sync_dist=True)
        return {"val_loss": loss, "val_psnr": psnr}

    def on_validation_epoch_end(self):
        if self.trainer.is_global_zero:
            train_loss = self.trainer.callback_metrics.get("train_loss", None)
            val_loss = self.trainer.callback_metrics.get("val_loss", None)
            val_psnr = self.trainer.callback_metrics.get("val_psnr", None)

            if (
                train_loss is not None and
                val_loss is not None and
                val_psnr is not None
            ):
                print(
                    f"\nEpoch {self.current_epoch}: "
                    f"train_loss={train_loss.item():.4f} "
                    f"val_loss={val_loss.item():.4f} "
                    f"val_psnr={val_psnr.item():.2f}")

    def _calculate_psnr(self, output, target):
        mse = torch.mean((output - target) ** 2)
        if mse == 0:
            return torch.tensor(100.0, device=output.device)
        return 20 * torch.log10(1.0 / torch.sqrt(mse))

    def perceptual_loss(self, pred, target):
        pred_norm = self.normalize(pred)
        target_norm = self.normalize(target)
        feat_pred = self.vgg(pred_norm)
        feat_target = self.vgg(target_norm)
        return torch.nn.functional.l1_loss(feat_pred, feat_target)

    def lr_scheduler_step(self, scheduler, metric):
        scheduler.step()
        # lr = scheduler.get_lr()

    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.parameters(), lr=opt.lr, weight_decay=1e-4)
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer=optimizer, warmup_epochs=15, max_epochs=opt.epochs)

        return [optimizer], [scheduler]


def main():
    print("Options")
    print(opt)
    if opt.wblogger is not None:
        logger = WandbLogger(project=opt.wblogger, name="PromptIR-Train")
    else:
        logger = TensorBoardLogger(save_dir="logs/")

    trainset = PromptTrainDataset(opt)
    trainloader = DataLoader(
        trainset, batch_size=opt.batch_size,
        pin_memory=True, shuffle=True,
        drop_last=True, num_workers=opt.num_workers)

    valset = PromptValDataset(opt)  # 你要自己建這個 Dataset
    valloader = DataLoader(
        valset, batch_size=opt.batch_size,
        shuffle=False, num_workers=opt.num_workers)

    model = PromptIRModel()

    trainer = pl.Trainer(
        max_epochs=opt.epochs, accelerator="gpu",
        devices=opt.num_gpus, strategy="ddp_find_unused_parameters_true",
        logger=logger, callbacks=[checkpoint_callback],
        enable_checkpointing=True)
    trainer.fit(
        model=model, train_dataloaders=trainloader,
        val_dataloaders=valloader)


if __name__ == '__main__':
    main()
