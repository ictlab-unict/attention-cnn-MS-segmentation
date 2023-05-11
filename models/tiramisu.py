import torch
import torch.nn as nn
import torch.nn.functional as F
from models import BiConvLSTM
from models import layers

class FCDenseNet(nn.Module):
    def __init__(self, in_channels=3, down_blocks=(5,5,5,5,5),
                 up_blocks=(5,5,5,5,5), bottleneck_layers=5,
                 growth_rate=16, out_chans_first_conv=48, loss_type = 'dice', n_classes=2, use_stn=False , use_sa = False, seq_size=1, use_lstm=False, block_to_cancel = -1, feature_to_cancel = -1, **lstm_parameters):
        super().__init__()

        self.down_blocks = down_blocks
        self.up_blocks = up_blocks
        self.seq_size = seq_size
        self.use_stn = use_stn
        self.use_lstm = use_lstm
        self.use_sa = use_sa
        self.block_to_cancel = block_to_cancel
        self.feature_to_cancel = feature_to_cancel
        cur_channels_count = 0
        skip_connection_channel_counts = []

        ## First Convolution ##

        self.add_module('firstconv', nn.Conv2d(in_channels=in_channels,
                  out_channels=out_chans_first_conv, kernel_size=3,
                  stride=1, padding=1, bias=True))
        cur_channels_count = out_chans_first_conv

        #####################
        # Downsampling path #
        #####################

        self.denseBlocksDown = nn.ModuleList([])
        self.saBlocksDown = nn.ModuleList([])
        self.transDownBlocks = nn.ModuleList([])

        for i in range(len(down_blocks)):
            self.denseBlocksDown.append(
                layers.DenseBlock(cur_channels_count, growth_rate, down_blocks[i]))
            cur_channels_count += (growth_rate * down_blocks[i])
            if self.use_sa:
                self.saBlocksDown.append(layers.SqueezeAttentionBlock(cur_channels_count, cur_channels_count))
            skip_connection_channel_counts.insert(0,cur_channels_count)
            self.transDownBlocks.append(layers.TransitionDown(cur_channels_count))

        #####################
        #     Bottleneck    #
        #####################

        self.add_module('bottleneck',layers.Bottleneck(cur_channels_count,
                                     growth_rate, bottleneck_layers))
        prev_block_channels = growth_rate*bottleneck_layers
        cur_channels_count += prev_block_channels

       
        #####################
        #    BI-DIR LSTM    #
        #####################
        if self.use_lstm:
            if lstm_parameters['lstm_parameters']['bidirectional'] == True:
                num_directions = 2
            else:
                num_directions = 1
            self.biConvLSTM = BiConvLSTM.BiConvLSTM(prev_block_channels, prev_block_channels//num_directions, lstm_parameters['lstm_parameters']['lstm_kernel_size'], lstm_parameters['lstm_parameters']['lstm_num_layers'],  lstm_parameters['lstm_parameters']['bidirectional'], True)


        #######################
        #   Upsampling path   #
        #######################

        self.transUpBlocks = nn.ModuleList([])
        self.denseBlocksUp = nn.ModuleList([])
        self.saBlocksUp = nn.ModuleList([])
        for i in range(len(up_blocks)-1):
            self.transUpBlocks.append(layers.TransitionUp(prev_block_channels, prev_block_channels))
            cur_channels_count = prev_block_channels + skip_connection_channel_counts[i]

            self.denseBlocksUp.append(layers.DenseBlock(
                cur_channels_count, growth_rate, up_blocks[i],
                    upsample=True))
            prev_block_channels = growth_rate*up_blocks[i]
            cur_channels_count += prev_block_channels
            if self.use_sa:
                self.saBlocksUp.append(layers.SqueezeAttentionBlock(prev_block_channels, prev_block_channels))

        ## Final DenseBlock ##

        self.transUpBlocks.append(layers.TransitionUp(
            prev_block_channels, prev_block_channels))
        cur_channels_count = prev_block_channels + skip_connection_channel_counts[-1]

        self.denseBlocksUp.append(layers.DenseBlock(
            cur_channels_count, growth_rate, up_blocks[-1],
                upsample=False))
        cur_channels_count += growth_rate*up_blocks[-1]

        ## Softmax ##

        self.finalConv = nn.Conv2d(in_channels=cur_channels_count,
               out_channels=n_classes, kernel_size=1, stride=1,
                   padding=0, bias=True)
        if loss_type == 'focal':
            self.softmax = nn.LogSoftmax(dim=1)
        else:
            self.softmax = nn.Softmax(dim=1)
        if self.use_stn:
        # Spatial transformer localization-network
            self.localization = nn.Sequential(
                nn.Conv2d(3, 8, kernel_size=7),
                nn.MaxPool2d(2, stride=2),
                nn.ReLU(True),
                nn.Conv2d(8, 10, kernel_size=5),
                nn.MaxPool2d(2, stride=2),
                nn.ReLU(True)
            )

            # Regressor for the 3 * 2 affine matrix
            self.fc_loc = nn.Sequential(
                nn.Linear(10 * 52 * 52, 32),
                nn.ReLU(True),
                nn.Linear(32, 3 * 2)
            )

            # Initialize the weights/bias with identity transformation
            self.fc_loc[2].weight.data.fill_(0)
            self.fc_loc[2].bias.data = torch.FloatTensor([1, 0, 0, 0, 1, 0])

    # Spatial transformer network forward function
    def stn(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 10 * 52 * 52)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)

        return x

    def forward(self, x):
        deep_out = []
        features_mean = []
        h = x.size(3)
        w = x.size(4)
        batch_size = x.size(0)
        x = x.view(batch_size*self.seq_size, x.size(2), h, w)
        
        if self.use_stn:
            out = self.stn(x)
            out = self.firstconv(out)
        else:
            out = self.firstconv(x)

        skip_connections = []
        for i in range(len(self.down_blocks)):
            out = self.denseBlocksDown[i](out)
            if self.use_sa:
                out = self.saBlocksDown[i](out)
            skip_connections.append(out)
            out = self.transDownBlocks[i](out)
            block_features_mean = []
            num_features = out.size(1)
            for f in range(num_features):
                block_features_mean.append(out[:,f,:,:].mean().item())
            features_mean.append(block_features_mean)

        out = self.bottleneck(out)

        if self.use_lstm:
            out = out.view(batch_size, self.seq_size, out.size(1), out.size(2), out.size(3))
            out = out.transpose(0,1).contiguous()
            out, state = self.biConvLSTM(out, None)
            out = out.transpose(0,1).contiguous()
            out = out.view(batch_size*self.seq_size, out.size(2), out.size(3), out.size(4))

        for i in range(len(self.up_blocks)):
            skip = skip_connections.pop()
            out = self.transUpBlocks[i](out, skip)
            out = self.denseBlocksUp[i](out)
            if self.use_sa and i < len(self.up_blocks)-1:
                out = self.saBlocksUp[i](out)

            block_features_mean = []
            num_features = out.size(1)
            for f in range(num_features):
                block_features_mean.append(out[:, f, :, :].mean().item())
            features_mean.append(block_features_mean)

            if i == self.block_to_cancel:
                out[:,self.feature_to_cancel,:,:] = out[:,self.feature_to_cancel,:,:]*0



        out = self.finalConv(out)
        out = self.softmax(out)
        return (out,deep_out,features_mean)


def FCDenseNet57(loss_type = 'dice', n_classes = 2, grow_rate = 12, use_stn=False, use_sa = False, seq_size=1 , use_lstm=False, block_to_cancel = -1, feature_to_cancel = -1, **lstm_parameters):
    return FCDenseNet(
        in_channels=3, down_blocks=(4, 4, 4, 4, 4),
        up_blocks=(4, 4, 4, 4, 4), bottleneck_layers=4,
        growth_rate=grow_rate, out_chans_first_conv=48, loss_type = loss_type, n_classes = n_classes, use_stn=use_stn, use_sa = use_sa, seq_size=seq_size, use_lstm=use_lstm, block_to_cancel=block_to_cancel, feature_to_cancel=feature_to_cancel, lstm_parameters = lstm_parameters)


def FCDenseNet67(loss_type = 'dice', n_classes = 2, grow_rate = 16, use_stn=False, use_sa = False, seq_size=1 , use_lstm=False, block_to_cancel = -1, feature_to_cancel = -1,  **lstm_parameters):
    return FCDenseNet(
        in_channels=3, down_blocks=(5, 5, 5, 5, 5),
        up_blocks=(5, 5, 5, 5, 5), bottleneck_layers=5,
        growth_rate=grow_rate, out_chans_first_conv=48, loss_type = loss_type, n_classes = n_classes, use_stn=use_stn, use_sa = use_sa, seq_size=seq_size, use_lstm=use_lstm, block_to_cancel=block_to_cancel, feature_to_cancel=feature_to_cancel, lstm_parameters = lstm_parameters)


def FCDenseNet103(loss_type = 'dice', n_classes = 2, grow_rate = 16, use_stn=False, use_sa = False, seq_size=1 , use_lstm=False, block_to_cancel = -1, feature_to_cancel = -1, **lstm_parameters):
    return FCDenseNet(
        in_channels=3, down_blocks=(4,5,7,10,12),
        up_blocks=(12,10,7,5,4), bottleneck_layers=15,
        growth_rate=grow_rate, out_chans_first_conv=40, loss_type = loss_type, n_classes= n_classes, use_stn=use_stn, use_sa = use_sa, seq_size=seq_size, use_lstm=use_lstm, block_to_cancel=block_to_cancel, feature_to_cancel=feature_to_cancel, lstm_parameters = lstm_parameters)
