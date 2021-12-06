from deeplearning.models import Model
from deeplearning.models.vision.pytorch.resnet import Resnet18
from deeplearning.models import Model
import torch
class AudioClassifier(Model):



    def __init__(self, num_classes, **kwargs):


        super().__init__(**kwargs)

        self.hidden_size = 512

        self.lstm = torch.nn.LSTM(input_size=20,
                    hidden_size=self.hidden_size,
                    num_layers=1,
                    batch_first=True,
                    bidirectional=True)

        self.fc = torch.nn.Linear(2*self.hidden_size, num_classes)

    def forward(self, x):

        features, lengths = x

        sort_idxs = torch.argsort(lengths, descending=True)
        original_idxs = torch.argsort(sort_idxs)

        features = torch.nn.utils.rnn.pack_padded_sequence(features[sort_idxs], lengths[sort_idxs].to(torch.device('cpu')), batch_first=True)

        hidden = (torch.randn(2, 1, self.hidden_size).to(x[0].device), torch.randn(2, 1, self.hidden_size).to(x[0].device))


        output, hidden = self.lstm(features)

        hidden = torch.swapaxes(hidden[0], 0, 1).reshape((hidden[0].shape[1], -1))

        output = self.fc(hidden)

        output = output[original_idxs]

        return output

    @staticmethod
    def args(parser):

        parser.add_argument('--num_classes', type=int)

        return super(AudioClassifier, AudioClassifier).args(parser)


        