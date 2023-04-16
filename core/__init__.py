from scipy.io import loadmat

from third_party.face_fitting_pytorch.core.BFM09Model import BFM09ReconModel


def get_recon_model(model="bfm09", **kargs):
    if model == "bfm09":
        model_path = "third_party/face_fitting_pytorch/BFM/BFM09_model_info.mat"
        model_dict = loadmat(model_path)
        recon_model = BFM09ReconModel(model_dict, **kargs)
        return recon_model
    else:
        raise NotImplementedError()
