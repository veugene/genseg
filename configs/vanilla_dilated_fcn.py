from architectures.image2image import DilatedFCN

def build_model():
    model_kwargs = {'in_channels': 4, 'C': 24, 'classes': 3}
    return DilatedFCN(**model_kwargs)
