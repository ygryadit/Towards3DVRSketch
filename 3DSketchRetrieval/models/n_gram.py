import torchvision.models as models
from .Model import Model
import torch
import torch.nn as nn
from config import PRETRAINED_PATH


class NGVNN(Model):
    def __init__(self, name, nclasses=40, pretraining=False, cnn_name='vgg11_bn', num_views=12):
        super(NGVNN, self).__init__(name)
        self.num_views = num_views
        if cnn_name == 'vgg11_bn':
            model = models.vgg11_bn(pretrained=False)
            if pretraining:
                model.load_state_dict(torch.load(PRETRAINED_PATH))
            self.features = model.features
            self.avgpool = model.avgpool
            self.classifier = nn.Sequential(*list(model.classifier.children())[:2])

        embedding_dim = 4096
        num_filters = 512
        self._ngram_filter_sizes = [3, 5, 7]
        self._convolution_layers = [nn.Conv1d(
            in_channels=embedding_dim,
            out_channels=num_filters,
            kernel_size=ngram_size) for ngram_size in self._ngram_filter_sizes]
        for i, conv_layer in enumerate(self._convolution_layers):
            self.add_module('conv_layer_%d' % i, conv_layer)

        self._activation = nn.ReLU()
        self._softmax = nn.Softmax(dim=1)
        self._layernorm = nn.LayerNorm(num_filters)

        self._fc1 = nn.Linear(num_filters * len(self._ngram_filter_sizes), num_filters)
        self._fc2 = nn.Linear(num_filters, nclasses)

    def forward(self, x, train=True):
        #feature extractor
        y = self.features(x)
        y = self.avgpool(y)
        y = self.classifier(torch.flatten(y, 1))
        y = y.view((int(x.shape[0]/self.num_views), -1, y.shape[-1])) #[n, V, D]

        # N-gram + attention
        filter_outputs = []
        for i in range(len(self._convolution_layers)):
            convolution_layer = getattr(self, 'conv_layer_{}'.format(i))
            token = self._activation(convolution_layer(y.transpose(1, 2)))# [n, D', num_gram]
            g_p = token.max(dim=2)[0]# [n, D']
            phi = torch.bmm(token.transpose(1, 2), g_p.unsqueeze(-1)) / (token.shape[1] ** .5)
            beta = self._softmax(phi)# [n, num_gram, 1]
            g_a = torch.bmm(beta.transpose(1, 2), token.transpose(1, 2))# [n, 1, D']
            g = g_a.squeeze(1) + g_p# [n, D']
            g = self._layernorm(g)
            filter_outputs.append(g)

        maxpool_output = torch.cat(filter_outputs, dim=1) if len(filter_outputs) > 1 else filter_outputs[0]
        feat = self._fc1(maxpool_output)
        if not train:
            return feat
        predict = self._fc2(feat)
        return predict, feat

