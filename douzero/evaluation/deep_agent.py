import numpy as np
import torch

from douzero.env.env import get_obs


def _safe_torch_load(model_path, device):
    try:
        return torch.load(model_path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(model_path, map_location=device)


def _load_model(position, model_path):
    from douzero.dmc.models import model_dict

    model = model_dict[position]()
    model_state_dict = model.state_dict()
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    pretrained = _safe_torch_load(model_path, device)
    pretrained = {k: v for k, v in pretrained.items() if k in model_state_dict}
    model_state_dict.update(pretrained)
    model.load_state_dict(model_state_dict)
    if torch.cuda.is_available():
        model.cuda()
    model.eval()
    return model


class DeepAgent:

    def __init__(self, position, model_path):
        self.model = _load_model(position, model_path)

    def act(self, infoset):
        # Always evaluate neural network forward pass even when only 1 legal action exists
        # so the caller receives an accurate win-rate confidence estimate.
        obs = get_obs(infoset)
        z_batch = torch.from_numpy(obs['z_batch']).float()
        x_batch = torch.from_numpy(obs['x_batch']).float()
        if torch.cuda.is_available():
            z_batch, x_batch = z_batch.cuda(), x_batch.cuda()
        with torch.no_grad():
            y_pred = self.model.forward(z_batch, x_batch, return_value=True)['values']
        y_pred = y_pred.detach().cpu().numpy()

        best_action_index = np.argmax(y_pred, axis=0)[0]
        best_action = infoset.legal_actions[best_action_index]
        best_action_confidence = y_pred[best_action_index]
        return best_action, best_action_confidence
