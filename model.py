import torch
import torch.nn as nn
import torch.nn.functional as F


class SEAttention(nn.Module):
    def __init__(self, in_channels):
        super(SEAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(in_channels, in_channels // 16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(in_channels // 16, in_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc1(y)
        y = self.relu(y)
        y = self.fc2(y)
        y = self.sigmoid(y).view(b, c, 1, 1)
        return x * y

    class FeatureReorganizationModule(nn.Module):
        def __init__(self, in_channels):
            super(FeatureReorganizationModule, self).__init__()
            self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
            self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=5, padding=2)
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
            self.max_pool = nn.AdaptiveMaxPool2d(1)
            self.softmax = nn.Softmax(dim=1)

        def forward(self, x):
            x = self.conv1(x)
            x = self.conv2(x)
            avg_pooled = self.avg_pool(x)
            max_pooled = self.max_pool(x)
            pooled_sum = avg_pooled + max_pooled
            attention_weights = self.softmax(pooled_sum.squeeze(-1).squeeze(-1)).unsqueeze(-1).unsqueeze(-1)
            output = x * attention_weights
            return output

    class InterlayerFeatureFusionModule(nn.Module):
        def __init__(self, shallow_channels, intermediate_channels, deep_channels):
            super(InterlayerFeatureFusionModule, self).__init__()
            self.shallow_conv1 = nn.Conv2d(shallow_channels, shallow_channels, kernel_size=3, padding=1)
            self.shallow_conv2 = nn.Conv2d(shallow_channels, shallow_channels, kernel_size=1)
            self.intermediate_conv = nn.Conv2d(intermediate_channels, intermediate_channels, kernel_size=1)
            self.deep_conv1 = nn.Conv2d(deep_channels, deep_channels, kernel_size=3, padding=1)
            self.deep_conv2 = nn.Conv2d(deep_channels, deep_channels, kernel_size=1)
            self.frm = FeatureReorganizationModule(shallow_channels + intermediate_channels + deep_channels)

        def forward(self, shallow_feature, intermediate_feature, deep_feature):

            shallow_feature = self.shallow_conv1(shallow_feature)
            shallow_feature = self.shallow_conv2(shallow_feature)

            intermediate_feature = self.intermediate_conv(intermediate_feature)
            deep_feature = self.deep_conv1(deep_feature)
            deep_feature = self.deep_conv2(deep_feature
            fused_feature = torch.cat([shallow_feature, intermediate_feature, deep_feature], dim=1)
            output = self.frm(fused_feature)
            return output
class GraphConvolution(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GraphConvolution, self).__init__()
        self.weight = nn.Parameter(torch.randn(in_channels, out_channels))

    def forward(self, x, adjacency_matrix):
        support = torch.matmul(x, self.weight)
        output = torch.matmul(adjacency_matrix, support)
        return output


class GraphFeatureComparisonModule(nn.Module):
    def __init__(self, channels):
        super(GraphFeatureComparisonModule, self).__init__()
        self.channels = channels
        self.se = SEAttention(channels)
        self.graph_conv1 = GraphConvolution(channels, channels)
        self.graph_conv2 = GraphConvolution(channels, channels)
        self.graph_conv3 = GraphConvolution(channels, channels)

        self.W_g = nn.Parameter(torch.randn(3, 1))

    def cosine_similarity(self, a, b):

        numerator = torch.sum(a * b, dim=-1)
        denominator = torch.norm(a, dim=-1) * torch.norm(b, dim=-1)
        return 1 - numerator / (denominator + 1e-8)

    def forward(self, feature_map1, feature_map2):

        nodes1 = feature_map1.view(feature_map1.size(0), self.channels, -1).permute(0, 2, 1)
        nodes2 = feature_map2.view(feature_map2.size(0), self.channels, -1).permute(0, 2, 1)

        co_activation_strength = torch.matmul(nodes1, nodes2.transpose(-2, -1))


        nodes1 = self.se(nodes1.permute(0, 2, 1)).permute(0, 2, 1)
        nodes2 = self.se(nodes2.permute(0, 2, 1)).permute(0, 2, 1)


        se_weights = self.se(feature_map1).squeeze(-1).squeeze(-1)
        scaling_factors = torch.ones_like(se_weights)  # 假设缩放因子为1，实际可调整
        combined_weights = torch.stack([se_weights, scaling_factors, se_weights * scaling_factors], dim=1)
        final_edges = torch.sigmoid(torch.matmul(self.W_g, combined_weights).squeeze(0))


        nodes1_conv1 = self.graph_conv1(nodes1, final_edges)
        nodes1_conv2 = self.graph_conv2(nodes1_conv1, final_edges)
        nodes1_conv3 = self.graph_conv3(nodes1_conv2, final_edges)


        consistency_loss = 0
        num_layers = 3
        for layer_idx in range(num_layers):
            layer_feature1 = nodes1_conv1 if layer_idx == 0 else nodes1_conv2 if layer_idx == 1 else nodes1_conv3
            layer_feature2 = nodes2 if layer_idx == 0 else nodes2 if layer_idx == 1 else nodes2
            layer_loss = 0
            for i in range(self.channels):
                for j in range(self.channels):
                    sim_value = self.cosine_similarity(layer_feature1[:, i, :], layer_feature2[:, j, :])
                    layer_loss += torch.mean(sim_value)
            layer_loss /= (self.channels * self.channels)
            consistency_loss += layer_loss
        consistency_loss /= num_layers

        return  consistency_loss



class ChannelAttentionFusionModule(nn.Module):
    def __init__(self, in_channels_list, out_channels):
        super(ChannelAttentionFusionModule, self).__init__()
        self.in_channels_list = in_channels_list
        self.out_channels = out_channels
        self.convs = nn.ModuleList()
        for in_channels in in_channels_list:
            self.convs.append(nn.Conv2d(in_channels, out_channels, kernel_size=1))

        self.split_channels = out_channels // 2


        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.channel_shuffle = nn.PixelShuffle(2)


        self.q_conv = nn.Conv2d(self.split_channels, self.split_channels, kernel_size=3, padding=1)
        self.k_conv = nn.Conv2d(self.split_channels, self.split_channels, kernel_size=3, padding=1)
        self.v_conv = nn.Conv2d(self.split_channels, self.split_channels, kernel_size=3, padding=1)

    def forward(self, feature_list):

        unified_features = []
        for i, feature in enumerate(feature_list):
            feature = self.convs[i](feature)
            unified_features.append(feature)

        summed_feature = sum(unified_features)


        split_feature_1, split_feature_2 = torch.chunk(summed_feature, 2, dim=1)

        intra_channel_weights = self.avg_pool(split_feature_1)
        shuffled_feature = self.channel_shuffle(split_feature_1)
        upper_branch_output = shuffled_feature * intra_channel_weights


        q = self.q_conv(split_feature_2)
        k = self.k_conv(split_feature_2)
        v = self.v_conv(split_feature_2)

        attn_map = torch.matmul(q, k.transpose(-2, -1))
        attn_map = torch.softmax(attn_map, dim=-1)
        lower_branch_output = torch.matmul(attn_map, v)


        final_feature = upper_branch_output + lower_branch_output
        return final_feature
