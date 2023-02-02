def create_model(load_path, upscale, modeltype):
    if modeltype == 'WDSR':
        from .SR_model import SRModel_WDSR as M
        m = M(load_path, upscale)
    elif modeltype == 'EDSR':
        from .SR_model import SRModel_EDSR as M
        m = M(load_path, upscale)
    elif modeltype == 'CNNSR':
        from .SR_model import SRCNN as M
        m = M(load_path, upscale)
    elif modeltype == 'WDSRRS':
        from .SR_model import SRModel_WDSRRS as M
        m = M(load_path, upscale)
    else:
        raise NotImplementedError('Generator model [{:s}] not recognized'.format(modeltype))
    return m
