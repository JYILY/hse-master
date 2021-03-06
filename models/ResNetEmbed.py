
import copy
import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision
from models.EmbedGuiding import EmbedGuiding


class ModifiedResNet(torchvision.models.resnet.ResNet):
    def __init__(self):
        # ResNet50
        super(ModifiedResNet, self).__init__(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3])

    def _forward_impl(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        return x


def make_trunk(pretrain=True):
    trunk = ModifiedResNet()
    if pretrain:
        state_dict = torch.hub.load_state_dict_from_url(torchvision.models.resnet.model_urls['resnet50'])
        trunk.load_state_dict(state_dict)
    return trunk


class ResNetEmbed(nn.Module):
    def __init__(self, cdict={}):
        super(ResNetEmbed, self).__init__()
        
        self.num_classes = cdict

        self.avgpool = nn.AvgPool2d(14, stride=1) # for 448*448 input

        self.softmax = nn.Softmax(dim=1)

        # I.    define the share part
        self.trunk = ModifiedResNet()
        state_dict = torch.hub.load_state_dict_from_url(torchvision.models.resnet.model_urls['resnet50'])
        self.trunk.load_state_dict(state_dict)

        # II.   define the L1(order) branch
        self.branch_L1 = copy.deepcopy(self.trunk.layer4)
        self.fc_L1 = nn.Linear(2048, self.num_classes['order'])

        # III.  define the L2(family) branch
        self.branch_L2_guide = copy.deepcopy(self.trunk.layer4)
        self.branch_L2_raw = copy.deepcopy(self.trunk.layer4)
        self.fc_L2_guide = nn.Linear(2048, self.num_classes['family'])
        self.fc_L2_raw = nn.Linear(2048, self.num_classes['family'])
        self.fc_L2_cat = nn.Linear(2048*2, self.num_classes['family'])

        # IV.   define the L3(genus) branch
        self.branch_L3_guide = copy.deepcopy(self.trunk.layer4)
        self.branch_L3_raw = copy.deepcopy(self.trunk.layer4)
        self.fc_L3_guide = nn.Linear(2048, self.num_classes['genus'])
        self.fc_L3_raw = nn.Linear(2048, self.num_classes['genus'])
        self.fc_L3_cat = nn.Linear(2048*2, self.num_classes['genus'])

        # V.    define the L4(class) branch
        self.branch_L4_guide = copy.deepcopy(self.trunk.layer4)
        self.branch_L4_raw = copy.deepcopy(self.trunk.layer4)
        self.fc_L4_guide = nn.Linear(2048, self.num_classes['class'])
        self.fc_L4_raw = nn.Linear(2048, self.num_classes['class'])
        self.fc_L4_cat = nn.Linear(2048*2, self.num_classes['class'])

        # VI.   define the guiding modules
        self.G12 = EmbedGuiding(prior='order', init=False)
        self.G23 = EmbedGuiding(prior='family', init=False)
        self.G34 = EmbedGuiding(prior='genus', init=True)

    def forward(self, x):
        bz = x.size(0) # batch size
        '''
        (1)     truck forwards
        '''
        f_share = self.trunk(x)

        '''
        (2)     L1 branch forwards and predict scores
        '''
        f_l1 = self.branch_L1(f_share)
        f_l1 = self.avgpool(f_l1)
        f_l1 = f_l1.view(bz, -1)
        s_l1 = self.fc_L1(f_l1)

        '''
        (3.1)   L2 branches forward, and one branch predicts directly
        '''
        f_l2_g = self.branch_L2_guide(f_share)
        f_l2_r = self.branch_L2_raw(f_share)
        f_l2_r = self.avgpool(f_l2_r)
        f_l2_r = f_l2_r.view(bz, -1)
        s_l2_r = self.fc_L2_raw(f_l2_r)

        '''
        (3.2)   L1 guides L2, 
        '''
        s_l1_ = Variable(s_l1.data.clone(), requires_grad=False).cuda()
        s_l1_ = self.softmax(s_l1_)
        f_12_g = self.G12(s_l1_, f_l2_g)
        f_12_g = torch.sum(f_12_g.view(f_12_g.size(0), f_12_g.size(1), -1), dim=2)
        f_12_g = f_12_g.view(bz, -1)
        s_l2_g = self.fc_L2_guide(f_12_g)

        '''
        (3.3)   predict scores of L2
        '''
        f_l2_cat = torch.cat((f_12_g, f_l2_r), dim=1)
        s_l2_cat = self.fc_L2_cat(f_l2_cat)

        s_l2_avg = (s_l2_r + s_l2_g + s_l2_cat) / 3

        '''
        (4.1)     L3 branches forward, and one branch predicts directly
        '''
        f_l3_g = self.branch_L3_guide(f_share)
        f_l3_r = self.branch_L3_raw(f_share)
        f_l3_r = self.avgpool(f_l3_r)
        f_l3_r = f_l3_r.view(bz, -1)
        s_l3_r = self.fc_L3_raw(f_l3_r)

        '''
        (4.2)   L2 guides L3, 
        '''
        s_l2_ = Variable(s_l2_avg.data.clone(), requires_grad=False).cuda()
        s_l2_ = self.softmax(s_l2_)
        f_13_g = self.G23(s_l2_, f_l3_g)
        f_13_g = torch.sum(f_13_g.view(f_13_g.size(0), f_13_g.size(1), -1), dim=2)
        f_13_g = f_13_g.view(bz, -1)
        s_l3_g = self.fc_L3_guide(f_13_g)

        '''
        (4.3)   predict scores of L3
        '''
        f_l3_cat = torch.cat((f_13_g, f_l3_r), dim=1)
        s_l3_cat = self.fc_L3_cat(f_l3_cat)

        s_l3_avg = (s_l3_r + s_l3_g + s_l3_cat) / 3

        '''
        (5.1)     L4 branches forward, and one branch predicts directly
        '''
        f_l4_g = self.branch_L4_guide(f_share)
        f_l4_r = self.branch_L4_raw(f_share)
        f_l4_r = self.avgpool(f_l4_r)
        f_l4_r = f_l4_r.view(bz, -1)
        s_l4_r = self.fc_L4_raw(f_l4_r)

        '''
        (5.2)   L3 guides L4, 
        '''
        s_l3_ = Variable(s_l3_avg.data.clone(), requires_grad=False).cuda()
        s_l3_ = self.softmax(s_l3_)
        f_14_g = self.G34(s_l3_, f_l4_g)
        f_14_g = torch.sum(f_14_g.view(f_14_g.size(0), f_14_g.size(1), -1), dim=2)
        f_14_g = f_14_g.view(bz, -1)
        s_l4_g = self.fc_L4_guide(f_14_g)
        '''
        (5.3)   predict scores of L4
        '''
        f_l4_cat = torch.cat((f_14_g, f_l4_r), dim=1)
        s_l4_cat = self.fc_L4_cat(f_l4_cat)
        s_l4_avg = (s_l4_r + s_l4_g + s_l4_cat) / 3

        return s_l1, s_l2_avg, s_l3_avg, s_l4_avg

