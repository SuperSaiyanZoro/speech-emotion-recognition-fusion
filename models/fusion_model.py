import torch
import torch.nn as nn
import torch.nn.functional as F
from config import config

class FusionModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc_w2v = nn.Linear(config.wav2vec_dim, 256)
        self.fc_mfcc = nn.Linear(40 * 100, 256)
        self.fc_mel = nn.Linear(128 * 100, 256)
        self.fc_pros = nn.Linear(12, 64)

        self.fusion = nn.Linear(256 * 3 + 64, 128)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(128, config.num_classes)

    def forward(self, w2v, mfcc, mel, pros):

        w2v = F.relu(self.fc_w2v(w2v))
        mfcc = F.relu(self.fc_mfcc(mfcc.view(mfcc.size(0), -1)))
        mel = F.relu(self.fc_mel(mel.view(mel.size(0), -1)))
        pros = F.relu(self.fc_pros(pros))

        fused = torch.cat([w2v, mfcc, mel, pros], dim=1)
        fused = self.dropout(F.relu(self.fusion(fused)))

        return self.classifier(fused)
