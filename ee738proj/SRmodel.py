import torch
import torch.nn as nn
import torchaudio
import math

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class SpeechRecognitionModel_Refined2(nn.Module):

    def __init__(self, n_classes=11):
        super(SpeechRecognitionModel_Refined2, self).__init__()
        self.pe = PositionalEncoding(256)
        
        cnns = [nn.Dropout(0.1),  
                nn.Conv1d(40,64,3, stride=1, padding=1),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Dropout(0.1),  
                nn.Conv1d(64,128,3, stride=1, padding=1),
                nn.BatchNorm1d(128),
                nn.ReLU()] 

        # ## define CNN layers
        # self.cnns = nn.Sequential(*nn.ModuleList(cnns))

        self.cnns = nn.Sequential(*nn.ModuleList(cnns))
        
        self.Linear1 = nn.Linear(128, 256)
        
   
        self.CE_Block = torchaudio.models.Conformer(
            input_dim=256,
            num_heads=4,
            ffn_dim=512,
            num_layers=4,
            depthwise_conv_kernel_size=11,
            dropout=0.1
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(256, 1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, n_classes)
        )
        
        
        ## define RNN layers as self.lstm - use a 3-layer bidirectional LSTM with 256 output size and 0.1 dropout
        # < fill your code here >
        # self.lstm = nn.LSTM(input_size=64, hidden_size=256, num_layers=3, bidirectional=True, dropout=0.1)

        self.preprocess   = torchaudio.transforms.MFCC(sample_rate=8000, n_mfcc=40)
        self.instancenorm = nn.InstanceNorm1d(40)

    def forward(self, x):

        ## compute MFCC and perform mean variance normalisation
        with torch.no_grad():
          x = self.preprocess(x)+1e-6
          x = self.instancenorm(x).detach()

        ## pass the network through the CNN layers
        # < fill your code here >       # x = (N, C, T)
        x = self.cnns(x)
        x = x.permute(0,2,1)            # x = (N, T, C)
        x = self.Linear1(x)             # x = (N, T, 256)
        x = x.permute(1,0,2)            # x = (T, N, 256)
        x = self.pe(x)                  # x = (T, N, 256)
        x = x.permute(1,0,2)            # x = (N, T, 256)
        
        x, _ = self.CE_Block(x, torch.tensor([x.size(1)]).repeat(x.size(0)).cuda())


        ## pass the network through the RNN layers - check the input dimensions of nn.LSTM()
        # < fill your code here >
        x = x.permute(1,0,2)            # x = (T, N, 256)


        ## pass the network through the classifier
        # < fill your code here >
        x = self.classifier(x)
        x = torch.log_softmax(x, dim=-1)
        return x
    
    
class SpeechRecognitionModel_Refined3(nn.Module):

    def __init__(self, n_classes=11):
        super(SpeechRecognitionModel_Refined3, self).__init__()
        self.pe = PositionalEncoding(256)
        
        cnns = [nn.Dropout(0.1),  
                nn.Conv1d(40,64,3, stride=1, padding=1),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Dropout(0.1),  
                nn.Conv1d(64,128,3, stride=1, padding=1),
                nn.BatchNorm1d(128),
                nn.ReLU()] 

        # ## define CNN layers
        # self.cnns = nn.Sequential(*nn.ModuleList(cnns))

        self.cnns = nn.Sequential(*nn.ModuleList(cnns))
        
        self.Linear1 = nn.Linear(128, 256)
        
   
        self.CE_Block = torchaudio.models.Conformer(
            input_dim=256,
            num_heads=4,
            ffn_dim=512,
            num_layers=4,
            depthwise_conv_kernel_size=11,
            dropout=0.1
        )
        
        # self.CE_Block2 = torchaudio.models.Conformer(
        #     input_dim=256,
        #     num_heads=4,
        #     ffn_dim=512,
        #     num_layers=4,
        #     depthwise_conv_kernel_size=31,
        #     dropout=0.1
        # )
        
        self.T_Block = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=256, nhead=4, dim_feedforward=512, dropout=0.1),
            num_layers=4
        )

        # self.T_Block2 = nn.TransformerEncoder(
        #     nn.TransformerEncoderLayer(d_model=256, nhead=4, dim_feedforward=512, dropout=0.1),
        #     num_layers=8
        # )
        self.classifier = nn.Sequential(
            nn.Linear(256, n_classes)
        )
        
        
        ## define RNN layers as self.lstm - use a 3-layer bidirectional LSTM with 256 output size and 0.1 dropout
        # < fill your code here >
        # self.lstm = nn.LSTM(input_size=64, hidden_size=256, num_layers=3, bidirectional=True, dropout=0.1)

        self.preprocess   = torchaudio.transforms.MFCC(sample_rate=8000, n_mfcc=40)
        self.instancenorm = nn.InstanceNorm1d(40)

    def forward(self, x):

        ## compute MFCC and perform mean variance normalisation
        with torch.no_grad():
          x = self.preprocess(x)+1e-6
          x = self.instancenorm(x).detach()

        ## pass the network through the CNN layers
        # < fill your code here >       # x = (N, C, T)
        x = self.cnns(x)
        x = x.permute(0,2,1)            # x = (N, T, C)
        x = self.Linear1(x)             # x = (N, T, 256)
        x = x.permute(1,0,2)            # x = (T, N, 256)
        x = self.pe(x)                  # x = (T, N, 256)
        x = x.permute(1,0,2)            # x = (N, T, 256)
        
        x, _ = self.CE_Block(x, torch.tensor([x.size(1)]).repeat(x.size(0)).cuda())
        # x, _ = self.CE_Block2(x, torch.tensor([x.size(1)]).repeat(x.size(0)).cuda())

        ## pass the network through the RNN layers - check the input dimensions of nn.LSTM()
        # < fill your code here >
        x = x.permute(1,0,2)            # x = (T, N, 256)
        x = self.T_Block(x)             # x = (T, N, 256)


        ## pass the network through the classifier
        # < fill your code here >
        x = self.classifier(x)
        x = torch.log_softmax(x, dim=-1)
        return x
    
    
    


  
class SpeechRecognitionModel_Refined10(nn.Module):

    def __init__(self, n_classes=11):
        super(SpeechRecognitionModel_Refined10, self).__init__()
        self.pe = PositionalEncoding(256)
        
        cnns = [nn.Dropout(0.1),  
                nn.Conv1d(40,64,3, stride=1, padding=1),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Dropout(0.1),  
                nn.Conv1d(64,128,3, stride=1, padding=1),
                nn.BatchNorm1d(128),
                nn.ReLU()] 

        # ## define CNN layers
        # self.cnns = nn.Sequential(*nn.ModuleList(cnns))

        self.cnns = nn.Sequential(*nn.ModuleList(cnns))
        
        self.Linear1 = nn.Linear(128, 256)
        
   
        self.CE_Block = torchaudio.models.Conformer(
            input_dim=256,
            num_heads=4,
            ffn_dim=512,
            num_layers=8,
            depthwise_conv_kernel_size=31,
            dropout=0.1
        )
        
        # self.CE_Block2 = torchaudio.models.Conformer(
        #     input_dim=256,
        #     num_heads=4,
        #     ffn_dim=512,
        #     num_layers=4,
        #     depthwise_conv_kernel_size=31,
        #     dropout=0.1
        # )
        
        self.T_Block = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=256, nhead=4, dim_feedforward=512, dropout=0.1),
            num_layers=4
        )

        # self.T_Block2 = nn.TransformerEncoder(
        #     nn.TransformerEncoderLayer(d_model=256, nhead=4, dim_feedforward=512, dropout=0.1),
        #     num_layers=8
        # )
        self.classifier = nn.Sequential(
            nn.Linear(256, n_classes)
        )
        
        
        ## define RNN layers as self.lstm - use a 3-layer bidirectional LSTM with 256 output size and 0.1 dropout
        # < fill your code here >
        # self.lstm = nn.LSTM(input_size=64, hidden_size=256, num_layers=3, bidirectional=True, dropout=0.1)

        self.preprocess   = torchaudio.transforms.MFCC(sample_rate=8000, n_mfcc=40)
        self.instancenorm = nn.InstanceNorm1d(40)

    def forward(self, x):

        ## compute MFCC and perform mean variance normalisation
        with torch.no_grad():
          x = self.preprocess(x)+1e-6
          x = self.instancenorm(x).detach()

        ## pass the network through the CNN layers
        # < fill your code here >       # x = (N, C, T)
        x = self.cnns(x)
        x = x.permute(0,2,1)            # x = (N, T, C)
        x = self.Linear1(x)             # x = (N, T, 256)
        x = x.permute(1,0,2)            # x = (T, N, 256)
        x = self.pe(x)                  # x = (T, N, 256)
        x = x.permute(1,0,2)            # x = (N, T, 256)
        
        x, _ = self.CE_Block(x, torch.tensor([x.size(1)]).repeat(x.size(0)).cuda())
        # x, _ = self.CE_Block2(x, torch.tensor([x.size(1)]).repeat(x.size(0)).cuda())

        ## pass the network through the RNN layers - check the input dimensions of nn.LSTM()
        # < fill your code here >
        x = x.permute(1,0,2)            # x = (T, N, 256)
        x = self.T_Block(x)             # x = (T, N, 256)


        ## pass the network through the classifier
        # < fill your code here >
        x = self.classifier(x)
        x = torch.log_softmax(x, dim=-1)
        return x
    
    
    