import onnx
import torch
import onnxruntime

from albert_emb.config import MODEL_FOLDER


full_model_ref = MODEL_FOLDER.joinpath('albert-base-v2.onnx')

onnx_model = onnx.load(full_model_ref)
onnx.checker.check_model(onnx_model)

ort_session = onnxruntime.InferenceSession(full_model_ref.as_posix())


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


def test_onnx_model():
    batch = 10
    input_ids = torch.randint(1, 20000, (batch, 1), requires_grad=False)

    import numpy as np
    from albert_emb.nlp_model import model as torch_model
    torch_out = torch_model(input_ids)

    # compute ONNX Runtime output prediction
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(input_ids)}
    ort_outs = ort_session.run(None, ort_inputs)

    # compare ONNX Runtime and PyTorch results
    np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-05)


if __name__ == "__main__":
    test_onnx_model()
