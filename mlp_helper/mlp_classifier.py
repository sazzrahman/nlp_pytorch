import torch.nn as nn
import torch.nn.functional as F

class SurnameClassifier(nn.Module):
    # Was not tested independently
    # the call method will take care of the forward unit
    def __init__(self, input_dim, hidden_dim, output_dim):
        nn.Module.__init__(self)
        # super(SurnameClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x_in, apply_softmax=False):
        x_in = x_in.float() # necessary for input processing
        intermediate_vector = F.relu(self.fc1(x_in))
        prediction_vector = self.fc2(intermediate_vector)
        if apply_softmax:
            prediction_vector = F.softmax(prediction_vector, dim=1)
        return prediction_vector