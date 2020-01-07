import torch.nn as nn
import torch.nn.functional as F


class SurnameClassifier(nn.Module):
    # Was not tested independently
    # the call method will take care of the forward unit
    def __init__(self, initial_num_channels, num_classes, num_channels):
        nn.Module.__init__(self)
        # super(SurnameClassifier, self).__init__()
        self.convnet = nn.Sequential(
            nn.Conv1d(in_channels=initial_num_channels, out_channels=num_channels, kernel_size=3),
            nn.ELU(),
            nn.Conv1d(in_channels=num_channels, out_channels=num_channels, kernel_size=3, stride=2),
            nn.ELU(),
            nn.Conv1d(in_channels=num_channels, out_channels=num_channels, kernel_size=3, stride=2),
            nn.ELU(),
            nn.Conv1d(in_channels=num_channels, out_channels=num_channels, kernel_size=3),
            nn.ELU(),
        )
        self.fc = nn.Linear(num_channels, num_classes)

    def forward(self, x_in, apply_softmax=False):
        x_in = x_in.float()  # necessary for input processing
        features = self.convnet(x_in).squeeze(dim=2)
        prediction_vector = self.fc(features)
        if apply_softmax:
            prediction_vector = F.softmax(prediction_vector, dim=1)
        return prediction_vector
