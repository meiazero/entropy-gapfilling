"""Deep learning models for satellite image gap-filling.

Each model lives in its own subpackage and can be trained, evaluated,
and used for inference independently.

Subpackages:
    ae/           Convolutional Autoencoder
    vae/          Variational Autoencoder
    gan/          GAN with UNet generator + PatchGAN discriminator
    unet/         U-Net with skip connections and residual blocks
    transformer/  MAE-style Transformer

Shared infrastructure (shared/):
    base.py          BaseDLMethod abstract class (no pdi_pipeline dependency)
    dataset.py       InpaintingDataset - reads manifest CSV directly
    metrics.py       PSNR, SSIM, RMSE, compute_validation_metrics
    trainer.py       GapPixelLoss, EarlyStopping, TrainingHistory, checkpoints
    visualization.py Training curve plots
"""
