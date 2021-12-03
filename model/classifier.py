from torch import nn, Tensor
from omegaconf import DictConfig


class SimpleClassifier(nn.Module):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        padding = (cfg.model.kernel_size - 1) // 2
        self.pre_conv = nn.Conv2d(
            in_channels=cfg.model.input_channels,
            out_channels=cfg.model.hidden_channels,
            kernel_size=cfg.model.kernel_size,
            padding=padding,
            padding_mode="reflect",
        )
        self.conv1 = nn.Conv2d(
            in_channels=cfg.model.hidden_channels,
            out_channels=cfg.model.hidden_channels,
            kernel_size=cfg.model.kernel_size,
            padding=padding,
            padding_mode="reflect",
        )
        self.conv2 = nn.Conv2d(
            in_channels=cfg.model.hidden_channels,
            out_channels=cfg.model.hidden_channels,
            kernel_size=cfg.model.kernel_size,
            padding=padding,
            padding_mode="reflect",
        )
        self.conv3 = nn.Conv2d(
            in_channels=cfg.model.hidden_channels,
            out_channels=cfg.model.hidden_channels,
            kernel_size=cfg.model.kernel_size,
            padding=padding,
            padding_mode="reflect",
        )
        self.conv4 = nn.Conv2d(
            in_channels=cfg.model.hidden_channels,
            out_channels=cfg.model.hidden_channels,
            kernel_size=cfg.model.kernel_size,
            padding=padding,
            padding_mode="reflect",
        )
        self.conv5 = nn.Conv2d(
            in_channels=cfg.model.hidden_channels,
            out_channels=cfg.model.hidden_channels,
            kernel_size=cfg.model.kernel_size,
            padding=padding,
            padding_mode="reflect",
        )
        self.conv6 = nn.Conv2d(
            in_channels=cfg.model.hidden_channels,
            out_channels=cfg.model.hidden_channels,
            kernel_size=cfg.model.kernel_size,
            padding=padding,
            padding_mode="reflect",
        )
        self.conv7 = nn.Conv2d(
            in_channels=cfg.model.hidden_channels,
            out_channels=cfg.model.hidden_channels,
            kernel_size=cfg.model.kernel_size,
            padding=padding,
            padding_mode="reflect",
        )
        self.conv8 = nn.Conv2d(
            in_channels=cfg.model.hidden_channels,
            out_channels=cfg.model.hidden_channels,
            kernel_size=cfg.model.kernel_size,
            padding=padding,
            padding_mode="reflect",
        )
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.batchnorm = nn.BatchNorm2d(num_features=cfg.model.hidden_channels)
        self.linear = nn.Linear(in_features=cfg.model.hidden_channels * 49,
                                out_features=cfg.model.num_class)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()

    def forward(self, data: Tensor) -> Tensor:
        data = self.pre_conv(data)
        conv1 = self.conv1(self.relu(self.batchnorm(data)))
        conv2 = self.conv2(self.relu(self.batchnorm(conv1))) + data
        conv3 = self.conv3(self.relu(self.batchnorm(conv2)))
        conv4 = self.conv4(self.relu(self.batchnorm(conv3))) + conv2
        pool1 = self.max_pool(conv4)
        conv5 = self.conv5(self.relu(self.batchnorm(pool1)))
        conv6 = self.conv6(self.relu(self.batchnorm(conv5))) + pool1
        conv7 = self.conv7(self.relu(self.batchnorm(conv6)))
        conv8 = self.conv8(self.relu(self.batchnorm(conv7))) + conv6
        pool2 = self.max_pool(conv8)

        return self.linear(self.flatten(pool2))
