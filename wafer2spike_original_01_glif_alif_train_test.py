"""01) Original + GLIF/ALIF adaptive-threshold neurons."""
from wafer2spike_original_train_test import build_dataloaders, training, CurrentBasedSNN

if __name__ == "__main__":
    dataloaders = build_dataloaders()
    training(
        network=CurrentBasedSNN,
        params=[0.05, 0.10, 0.08, 0.30],
        dataloaders=dataloaders,
    )
