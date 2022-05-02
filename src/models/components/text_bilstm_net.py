import torch
import torch.nn as nn
import torch.nn.functional as F


class TextBiLSTMNet(nn.Module):
    def __init__(
        self,
        num_classes: int = 2,
        dropout: float = 0.5,
        num_layers: int = 2,
        hidden_size: int = 128,
        embed_size: int = 1024,
        bidirectional: bool = True,
    ):
        super(TextBiLSTMNet, self).__init__()

        self.attention_layer = nn.Sequential(
            nn.Linear(in_features=hidden_size, out_features=hidden_size), nn.ReLU(inplace=True)
        )

        self.lstm_layer = nn.LSTM(
            input_size=embed_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
            batch_first=True,
        )

        self.linear_layer = nn.Sequential(
            nn.Linear(in_features=hidden_size, out_features=hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(in_features=hidden_size, out_features=num_classes),
            nn.Softmax(dim=1),
        )

    def attention_with_w(self, lstm_out, lstm_hidden):
        lstm_out_chunks = torch.chunk(lstm_out, chunks=2, dim=-1)
        h = lstm_out_chunks[0] + lstm_out_chunks[1]

        lstm_hidden = torch.sum(lstm_hidden, dim=1, keepdims=True)
        attn_weights = self.attention_layer(lstm_hidden)

        m = nn.Tanh()(h)
        context = attn_weights @ m.transpose(1, 2)
        softmax_weights = F.softmax(context, dim=-1)

        context = (softmax_weights @ h).squeeze(dim=1)
        return context

    def forward(self, x):
        x, (h, _) = self.lstm_layer(x)
        attn = self.attention_with_w(lstm_out=x, lstm_hidden=h.permute(1, 0, 2))
        y = self.linear_layer(attn)
        return y


if __name__ == "__main__":
    model = TextBiLSTMNet()
    x = torch.randn((10, 3, 1024))
    y = model(x)
    print(" input_size:{}\noutput_size:{}".format(x.shape, y.shape))
