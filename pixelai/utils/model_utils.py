def count_parameters(model):
    """ Count total number of parameters of the provided model """
    return sum(p.numel() for p in model.parameters())

def count_trainable_parameters(model):
    """ Count total number of trainable parameters of the provided model """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)