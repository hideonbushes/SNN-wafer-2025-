"""01) Improved (GroupNorm + FastSigmoid) + GLIF/ALIF adaptive-threshold neurons."""
from wafer2spike_improved_train_test import build_dataloaders, training, CurrentBasedSNN

if __name__ == "__main__":
    dataloaders = build_dataloaders()
    training(
        network=CurrentBasedSNN,
        params=[0.05, 0.10, 0.08, 5.0],
        dataloaders=dataloaders,
        spike_ts=10,
        batch_size=256,
        epochs=20,
        lr=1e-4,
        dropout_fc=0.20,
    )
