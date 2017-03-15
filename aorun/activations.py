from torch.nn.functional import relu, softmax


def get(obj):
    if callable(obj):
        return obj
    elif type(obj) is str:
        if obj in globals():
            return globals()[obj]
        else:
            raise Exception(f'Unknown activation: {obj}')
    else:
        raise Exception('Activation must be a callable or str')
