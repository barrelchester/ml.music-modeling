from deeplearning.models import Model
import torch
class AudioClassifier(Model):



    def __init__(self, num_classes, **kwargs):

        super().__init__(**kwargs)

        self.lstm = torch.nn.LSTM(input_size=20,
                    hidden_size=128,
                    num_layers=1,
                    batch_first=True,
                    bidirectional=True)

        self.drop = torch.nn.Dropout(p=0.5)

        self.fc = torch.nn.Linear(2*128, num_classes)

    def forward(self, x):

        features, lengths = x

        sort_idxs = torch.argsort(lengths, descending=True)

        features = torch.nn.utils.rnn.pack_padded_sequence(features[sort_idxs], lengths[sort_idxs].to(torch.device('cpu')), batch_first=True)

        hidden = (torch.randn(2, 1, 128).to(x[0].device), torch.randn(2, 1, 128).to(x[0].device))

        output, hidden = self.lstm(features)

        hidden = torch.swapaxes(hidden[0], 0, 1).reshape((hidden[0].shape[1], -1))

        output = self.fc(hidden)

        return output

    @staticmethod
    def args(parser):

        parser.add_argument('--num_classes', type=int)

        return super(AudioClassifier, AudioClassifier).args(parser)


        