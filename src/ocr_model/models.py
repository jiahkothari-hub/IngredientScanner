"""Three text-line recognisers with the same input and output, so they can be compared fairly.

Input : grey line image, batch x 1 x 32 x W (W a multiple of 4)
Output: per-time-step class scores, T x batch x NUM_CLASSES with T = W / 4 (log-probabilities for CTC)

  ann   fully connected network (MLP). At every time step it sees a 32 x 24 pixel window, flattened into
        768 numbers. No weight sharing over space inside the window: a letter shifted by a few pixels
        is a different input pattern.
  cnn   convolutional network. Small 3 x 3 filters are shared over the whole image, so a letter shape is
        recognised wherever it is. Each output column is predicted on its own (local receptive field).
  crnn  the same CNN, then a 2-layer bidirectional LSTM that reads the column features left-to-right
        and right-to-left, so every prediction can use the whole line as context.
"""
import torch
import torch.nn.functional as F
from torch import nn

from src.ocr_model.charset import NUM_CLASSES
from src.ocr_model.render_lines import HEIGHT, STRIDE


class ANN(nn.Module):
    WINDOW = 24

    def __init__(self, hidden=(1024, 512), dropout=0.2, num_classes=NUM_CLASSES):
        super().__init__()
        layers, size = [], HEIGHT * self.WINDOW
        for h in hidden:
            layers += [nn.Linear(size, h), nn.ReLU(), nn.Dropout(dropout)]
            size = h
        self.mlp = nn.Sequential(*layers, nn.Linear(size, num_classes))

    def forward(self, x):
        pad = (self.WINDOW - STRIDE) // 2
        x = F.pad(x, (pad, pad, 0, 0), mode="replicate")
        windows = x.unfold(3, self.WINDOW, STRIDE)               # B x 1 x 32 x T x 24
        b, _, h, t, w = windows.shape
        windows = windows.permute(0, 3, 1, 2, 4).reshape(b, t, h * w)
        return self.mlp(windows).log_softmax(-1).permute(1, 0, 2)


def conv_block(c_in, c_out, kernel=3, padding=1):
    return [nn.Conv2d(c_in, c_out, kernel, padding=padding, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True)]


class Backbone(nn.Module):
    """32 x W image -> 256 features per column, W / 4 columns. Height is pooled away step by step.
    Channel widths are kept small (32 -> 128) so that training fits on a laptop CPU."""
    OUT = 256

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            *conv_block(1, 32), nn.MaxPool2d(2, 2),                       # 16 x W/2
            *conv_block(32, 64), nn.MaxPool2d(2, 2),                      # 8 x W/4
            *conv_block(64, 128), *conv_block(128, 128), nn.MaxPool2d((2, 1), (2, 1)),    # 4 x W/4
            *conv_block(128, 128), nn.MaxPool2d((2, 1), (2, 1)),          # 2 x W/4
            *conv_block(128, self.OUT, kernel=(2, 1), padding=0),         # 1 x W/4
        )

    def forward(self, x):
        return self.net(x).squeeze(2).permute(0, 2, 1)                    # B x T x 256


class CNN(nn.Module):
    def __init__(self, dropout=0.2, num_classes=NUM_CLASSES):
        super().__init__()
        self.backbone = Backbone()
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(Backbone.OUT, num_classes))

    def forward(self, x):
        return self.head(self.backbone(x)).log_softmax(-1).permute(1, 0, 2)


class CRNN(nn.Module):
    def __init__(self, hidden=128, layers=2, dropout=0.2, num_classes=NUM_CLASSES):
        super().__init__()
        self.backbone = Backbone()
        self.rnn = nn.LSTM(Backbone.OUT, hidden, num_layers=layers, bidirectional=True, batch_first=True,
                           dropout=dropout if layers > 1 else 0)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(2 * hidden, num_classes))

    def forward(self, x):
        features, _ = self.rnn(self.backbone(x))
        return self.head(features).log_softmax(-1).permute(1, 0, 2)


MODELS = {"ann": ANN, "cnn": CNN, "crnn": CRNN}


def build(name: str) -> nn.Module:
    return MODELS[name]()


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def greedy_decode(log_probs: torch.Tensor, lengths=None) -> list:
    """Best class per time step, then CTC collapse: merge repeats, drop blanks. T x B x C -> list of strings."""
    from src.ocr_model.charset import BLANK, decode_ids
    best = log_probs.argmax(-1).permute(1, 0).tolist()
    texts = []
    for i, seq in enumerate(best):
        seq = seq[:int(lengths[i])] if lengths is not None else seq
        ids, previous = [], BLANK
        for c in seq:
            if c != previous and c != BLANK:
                ids.append(c)
            previous = c
        texts.append(decode_ids(ids))
    return texts
