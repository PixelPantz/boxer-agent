import time
import torch
import torchvision

model = torchvision.models.shufflenet_v2_x0_5(weights='IMAGENET1K_V1')
model.fc = torch.nn.Linear(model.fc.in_features, 18)


class StupidAgent(torch.nn.Module):
    def __init__(self, device):
        super(StupidAgent, self).__init__()
        self.model = model
        self.device = device
        self.softmax = torch.nn.Softmax()

    def __call__(self, state):
        state = state.to(self.device)
        image = state.permute(2, 0, 1)
        image = image / 255.
        image = torch.unsqueeze(image[:, 30:-30, 30:-30], dim=0)
        image = self.model(image)
        image = torch.flatten(image)
        return self.softmax(image) + 0.2 / 18


def load_module(model_path, device):
    agent = StupidAgent(device)
    agent.load_state_dict(torch.load(model_path))
    agent.to(device)
    return agent