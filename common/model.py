# 2023.0420.0253 @Brian

import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTM_SR(nn.Module):

    def __init__(self, input_feature_dim, hidden_feature_dim, num_layers, batch_size, num_classes, dropout):

        super(LSTM_SR, self).__init__()
        self.input_feature_dim = input_feature_dim
        self.hidden_feature_dim = hidden_feature_dim
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.num_classes = num_classes

        self.lstm = nn.LSTM(input_size=input_feature_dim, hidden_size=hidden_feature_dim, num_layers=num_layers, dropout=dropout)
        self.linear = nn.Linear(hidden_feature_dim, num_classes)
        self.hidden = self.init_hidden() # (hidden state, cell state)

        self.relu = nn.ReLU()  
        self.dropout = nn.Dropout(0.2)
        

    def init_hidden(self):

        return (torch.zeros(self.hidden_layer_num, self.batch_size, self.hidden_dim).cuda(),
                torch.zeros(self.hidden_layer_num, self.batch_size, self.hidden_dim).cuda())
    
        
    def forward(self, input):

        input = input.view(len(input), self.batch_size, -1)
        output, self.hidden = self.lstm(output, self.hidden)
        output = self.linear(output[-1].view(self.batch_size, -1))

        return output
    

class CNN_SR(nn.Module):

    def __init__(self, num_classes):

        super(CNN_SR, self).__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=3, stride=1) # (16, 1, 17 * 2 * window)
        # self.norm1 = nn.LayerNorm([16, 1, ]) # [B, H, W]
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, stride=1) # (32, 1, )
        self.conv3 = nn.Conv1d(32, 64, kernel_size=3, stride=1) # (64, 1, )
        self.conv4 = nn.Conv1d(64, 64, kernel_size=4, stride=2) # (64, 1, )
        # self.norm2 = nn.LayerNorm([64, 1, ])
        self.conv5 = nn.Conv1d(64, 128, kernel_size=4, stride=2) # (128, 1, )
        self.conv6 = nn.Conv1d(128, 128, kernel_size=4, stride=2) # (128, 1, )
        self.fc1 = nn.Linear(7808, 64)
        self.fc2 = nn.Linear(64, num_classes)

        self.relu = nn.ReLU()           
        self.dropout = nn.Dropout(0.25)

        
    def forward(self, x):

        output = self.relu(self.conv1(x)) 
        # output = self.norm1(self.relu(self.conv1(x)))
        output = self.relu(self.conv2(output))
        output = self.relu(self.conv3(output))
        output = self.relu(self.conv4(output)) 
        # output = self.norm2(self.relu(self.conv4(output)))
        output = self.relu(self.conv5(output))
        output = self.dropout(self.relu(self.conv6(output)))
        output = torch.flatten(output, 1) # flatten all dimensions except batch
        output = self.fc1(output)
        output = self.fc2(output)

        return output
