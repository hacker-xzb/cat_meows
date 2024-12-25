import torch
import torch.nn as nn

class CatSoundClassifier(nn.Module):
    def __init__(self, input_channels, hidden_size=64, num_classes=3):
        super(CatSoundClassifier, self).__init__()

        # self.feature_layer = nn.Linear(81, input_channels)
        
        # CNN层用于特征提取
        self.conv1 = nn.Conv1d(input_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(0.3)
        
        # LSTM层用于序列建模
        self.lstm = nn.LSTM(64, hidden_size, num_layers=2, batch_first=True, bidirectional=True)
        
        # 全连接层用于分类
        self.fc = nn.Linear(hidden_size * 2, num_classes)
        
    def forward(self, x):
        # CNN特征提取
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        x = self.dropout(x)
        
        # 准备LSTM输入
        x = x.permute(0, 2, 1)  # 将形状从[batch, channels, time]改为[batch, time, channels]
        
        # LSTM处理
        lstm_out, _ = self.lstm(x)
        
        # 取最后一个时间步的输出
        last_output = lstm_out[:, -1, :]
        
        # 分类
        output = self.fc(last_output)
        return output
