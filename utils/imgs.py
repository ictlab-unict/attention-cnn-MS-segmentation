def normalize(tensor):
    m1 = tensor.min()
    m2 = tensor.max()
    tensor = (tensor-m1)/(m2-m1)
    return tensor