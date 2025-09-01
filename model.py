import torch.nn as nn

class LSTMForecaster(nn.Module):
    def __init__(self, num_features, hidden_size, num_layers, output_size):
        super(LSTMForecaster,self).__init__()

        self.num_features = num_features
        self.hidden_size = hidden_size
        self.num_layer = num_layers
        self.output_size = output_size

        self.lstm = nn.LSTM(
            input_size=num_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self,x):
        lstm_out,(h_n,c_n) = self.lstm(x)
        out = lstm_out[:,-1,:]  # Getting the output for the final step.
        out = self.fc(out)
        out = out.view(x.size(0),-1,2) #changing the shape of the output to (batch_size,horizen,targets)
        return out
    

class CNN_LSTMForecaster(nn.Module):
    def __init__(self, num_features, hidden_size, num_layers, output_size, cnn_filters=64, kernel_size=7):
        super(CNN_LSTMForecaster, self).__init__()
        
        self.num_features = num_features
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.cnn_filters = cnn_filters
        self.kernel_size = kernel_size

        self.cnn = nn.Sequential(
            # Input shape: (batch_size, num_features, lookback_window)
            nn.Conv1d(
                in_channels=num_features, 
                out_channels=self.cnn_filters, 
                kernel_size=self.kernel_size,
                padding='same'
            ),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )

        self.lstm = nn.LSTM(
            input_size=self.cnn_filters,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True
        )

        self.fc = nn.Linear(self.hidden_size, self.output_size)

    def forward(self,x):

        #shape of x is (batch_size, lookback_window, num_features)
        x = x.permute(0, 2, 1) # for the convolution we need shape of x to be (batch_size, num_features, lookback_window)
        cnn_out = self.cnn(x)
        cnn_out = cnn_out.permute(0, 2, 1) #changing the shape back for LSTM
        lstm_out, (h_n, c_n) = self.lstm(cnn_out)
        out = lstm_out[:,-1,:]  # Getting the output for the final step.
        out = self.fc(out)
        out = out.view(x.size(0), -1, 2)
        
        return out