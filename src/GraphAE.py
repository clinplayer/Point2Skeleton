import torch
import torch.nn as nn
import torch.nn.functional as F
from GraphConv import GraphConv

class LinkPredNet(nn.Module):
    def __init__(self):
        super(LinkPredNet, self).__init__()

        self.conv0 = nn.Conv1d(in_channels=512, out_channels=16, kernel_size=1, bias=False)
        self.bn0 = nn.BatchNorm1d(16, track_running_stats=False)

        self.conv1 = GraphConv(in_features=20, out_features=32)
        self.bn1 = nn.BatchNorm1d(32, track_running_stats=False)

        self.conv2 = GraphConv(in_features=32, out_features=32)
        self.bn2 = nn.BatchNorm1d(32, track_running_stats=False)

        self.conv3 = GraphConv(in_features=32, out_features=48)
        self.bn3 = nn.BatchNorm1d(48, track_running_stats=False)

        self.conv4 = GraphConv(in_features=48, out_features=64)
        self.bn4 = nn.BatchNorm1d(64, track_running_stats=False)

        self.conv5 = GraphConv(in_features=64, out_features=64)
        self.bn5 = nn.BatchNorm1d(64, track_running_stats=False)

        self.conv6 = GraphConv(in_features=64, out_features=80)
        self.bn6 = nn.BatchNorm1d(80, track_running_stats=False)

        self.conv7 = GraphConv(in_features=80, out_features=96)
        self.bn7 = nn.BatchNorm1d(96, track_running_stats=False)

        self.conv8 = GraphConv(in_features=96, out_features=96)
        self.bn8 = nn.BatchNorm1d(96, track_running_stats=False)

        self.conv9 = GraphConv(in_features=96, out_features=102)
        self.bn9 = nn.BatchNorm1d(102, track_running_stats=False)

        self.conv10 = GraphConv(in_features=102, out_features=128)
        self.bn10 = nn.BatchNorm1d(128, track_running_stats=False)

        self.conv11 = GraphConv(in_features=128, out_features=128)
        self.bn11 = nn.BatchNorm1d(128, track_running_stats=False)

        self.conv12 = GraphConv(in_features=128, out_features=144)

        self.align_2_3 = nn.Sequential(nn.Conv1d(in_channels=32, out_channels=48, kernel_size=1, bias=False),
                                  nn.BatchNorm1d(48, track_running_stats=False))
        self.align_3_4 = nn.Sequential(nn.Conv1d(in_channels=48, out_channels=64, kernel_size=1, bias=False),
                                  nn.BatchNorm1d(64, track_running_stats=False))
        self.align_5_6 = nn.Sequential(nn.Conv1d(in_channels=64, out_channels=80, kernel_size=1, bias=False),
                                  nn.BatchNorm1d(80, track_running_stats=False))
        self.align_6_7 = nn.Sequential(nn.Conv1d(in_channels=80, out_channels=96, kernel_size=1, bias=False),
                                  nn.BatchNorm1d(96, track_running_stats=False))
        self.align_8_9 = nn.Sequential(nn.Conv1d(in_channels=96, out_channels=102, kernel_size=1, bias=False),
                                  nn.BatchNorm1d(102, track_running_stats=False))
        self.align_9_10 = nn.Sequential(nn.Conv1d(in_channels=102, out_channels=128, kernel_size=1, bias=False),
                                  nn.BatchNorm1d(128, track_running_stats=False))

        self.dc = InnerProductDecoder(dropout=0.0, act=lambda x: x)


    def encode(self, x, A):
        x = x.transpose(1, 2)
        x_shape = x[:, 0:512, :]
        x_skel = x[:, 512:516, :]

        x_shape = self.bn0(F.relu(self.conv0(x_shape)))
        x = torch.cat([x_shape, x_skel], 1)

        #20->32
        x1 = self.bn1(self.conv1(x, A))
        x1 = F.relu(x1)
        #32->32
        x2 = self.bn2(self.conv2(x1, A))
        x2 = F.relu(x2)
        #32->48
        x3 = self.bn3(self.conv3(x2, A)) + self.align_2_3(x2)
        x3 = F.relu(x3)
        #48->64
        x4 = self.bn4(self.conv4(x3,A)) + self.align_3_4(x3)
        x4 = F.relu(x4)
        #64->64
        x5 = self.bn5(self.conv5(x4, A)) + x4
        x5 = F.relu(x5)
        #64->80
        x6 = self.bn6(self.conv6(x5, A)) + self.align_5_6(x5)
        x6 = F.relu(x6)
        #80->96
        x7 = self.bn7(self.conv7(x6, A)) + self.align_6_7(x6)
        x7 = F.relu(x7)
        #96->96
        x8 = self.bn8(self.conv8(x7,A)) + x7
        x8 = F.relu(x8)
        #96->102
        x9 = self.bn9(self.conv9(x8,A)) + self.align_8_9(x8)
        x9 = F.relu(x9)
        #102->128
        x10 = self.bn10(self.conv10(x9,A)) + self.align_9_10(x9)
        x10 = F.relu(x10)
        #128->128
        x11 = self.bn11(self.conv11(x10,A)) + x10
        x11 = F.relu(x11)
        #128->144
        x12 = self.conv12(x11,A)

        return x12


    def forward(self, x, A):
        feat = self.encode(x, A)
        A = self.dc(feat)

        return A

    def recover_A(self, A_raw, A_mask, t=0.8):
        A_raw_bin = torch.gt(A_raw, t).float()
        A_recover = (A_raw_bin * A_mask).float()

        return A_recover

    def compute_loss(self, A_recon, A_input, known_mask):
        known_nodes = known_mask.sum()
        pos_weight = float(known_nodes - A_input.sum()) / A_input.sum()
        loss_MBCE = F.binary_cross_entropy_with_logits(A_recon, A_input, pos_weight=pos_weight, weight=known_mask)

        return loss_MBCE


class InnerProductDecoder(nn.Module):

    def __init__(self, dropout, act=torch.sigmoid):
        super(InnerProductDecoder, self).__init__()
        self.dropout = dropout
        self.act = act

    def forward(self, z):
        # z = F.dropout(z, self.dropout, training=self.training)
        adj = self.act(torch.bmm(z.transpose(1, 2), z))
        return adj
