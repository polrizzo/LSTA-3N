from torch.autograd import Variable
from model.LSTA.conv_LSTA_cell import *


class attention_model(nn.Module):
    def __init__(self, num_classes=8, mem_size=512, c_cam_classes=1000, is_with_ta3n=False):
        super(attention_model, self).__init__()
        self.num_classes = num_classes
        self.mem_size = mem_size
        self.lsta_cell = ConvLSTACell(2048, mem_size, c_cam_classes)
        self.avgpool = nn.AvgPool2d(7)
        self.dropout = nn.Dropout(0.7)
        self.fc = nn.Linear(mem_size, self.num_classes)
        self.classifier = nn.Sequential(self.dropout, self.fc)

        # static params use from external methods
        self.is_with_ta3n = is_with_ta3n
        self.loss_fn = None
        self.optimizer_fn = None
        self.optim_scheduler = None

    def forward(self, features, device):
        state_att = (Variable(torch.zeros(features.size(1), 1, 7, 7).to(device)),
                     Variable(torch.zeros(features.size(1), 1, 7, 7).to(device)))
        state_inp = (Variable(torch.zeros((features.size(1), self.mem_size, 7, 7)).to(device)),
                     Variable(torch.zeros((features.size(1), self.mem_size, 7, 7)).to(device)))

        state_inp_stack = []
        for t in range(features.size(0)):
            features_reshaped = features[t, :, :, :, :]
            state_att, state_inp, _ = self.lsta_cell(features_reshaped, state_att, state_inp)
            state_inp_stack.append(self.avgpool(state_inp[0]).view(state_inp[0].size(0), -1))

        feats = state_inp_stack[-1]
        logits = self.classifier(feats)
        if self.is_with_ta3n:  # In order to mantains featurs as TA3N wants
            feats = torch.stack(state_inp_stack, dim=1)
        return logits, feats

    # General utils
    def set_loss_fn(self, loss):
        self.loss_fn = loss

    def set_optimizer_fn(self, optimizer):
        self.optimizer_fn = optimizer

    def set_optim_scheduler(self, optim_scheduler):
        self.optim_scheduler = optim_scheduler
