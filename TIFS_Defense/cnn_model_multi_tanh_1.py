import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU
set_seed(42)  # Call the seed function with your desired seed

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        # nn.init.kaiming_uniform_(m.weight,nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)

class BottomModel(nn.Module):
    def __init__(self,gpu=False):
        super(BottomModel,self).__init__()
        self.gpu = gpu

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.apply(weights_init)

    def forward(self,x):
        x = x.to(device)

        x = F.relu(self.conv1(x)) 
        x = self.max_pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x)) 
        x = self.max_pool(F.relu(self.conv4(x))) 

        return x

class PredictModel(nn.Module):
    def __init__(self, gpu=False):
        super(PredictModel, self).__init__()
        self.gpu = gpu
        self.fc1 = nn.Linear(64*4*8, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.fc3 = nn.Linear(512, 256)
        self.bn3 = nn.BatchNorm1d(256)
        self.fc4 = nn.Linear(256, 128)
        self.bn4 = nn.BatchNorm1d(128)
        self.fc5 = nn.Linear(128, 64)
        self.bn5 = nn.BatchNorm1d(64)
        self.fc6 = nn.Linear(64, 128)
        self.bn6 = nn.BatchNorm1d(128)
        self.fc7 = nn.Linear(128, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.fc8 = nn.Linear(256, 512)
        self.bn8 = nn.BatchNorm1d(512)
        self.fc9 = nn.Linear(512, 1024)
        self.bn9 = nn.BatchNorm1d(1024)
        self.fc10 = nn.Linear(1024, 64*4*8)
        self.dropout = nn.Dropout(0.5)

        self.apply(weights_init)

    def forward(self, x):
        x = x.to(device)

        x = x.view(x.size(0), -1)
        # print(f"PredictModel input shape: {x.shape}")
        ## 調整成 tanh 看看
        x = F.tanh(self.bn1(self.fc1(x)))
        x = F.tanh(self.bn2(self.fc2(x)))
        x = F.tanh(self.bn3(self.fc3(x)))
        x = F.tanh(self.bn4(self.fc4(x)))
        # x = F.tanh(self.fc4(x))
        x = F.tanh(self.bn5(self.fc5(x)))
        # x = F.tanh(self.fc5(x))
        x = F.tanh(self.bn6(self.fc6(x)))
        # x = F.tanh(self.fc6(x))
        x = F.tanh(self.bn7(self.fc7(x)))
        x = F.tanh(self.bn8(self.fc8(x)))
        x = F.tanh(self.bn9(self.fc9(x)))
        # x = F.tanh(self.fc7(x))
        # x = F.tanh(self.fc8(x))
        # x = F.tanh(self.fc9(x))
        # x = F.tanh(self.bn1(self.fc1(x)))
        # x = self.dropout(x)
        # x = F.tanh(self.bn2(self.fc2(x)))
        # # x = self.dropout(x)
        # x = F.tanh(self.bn3(self.fc3(x)))
        # # x = self.dropout(x)
        # x = F.tanh(self.bn4(self.fc4(x)))
        # # x = self.dropout(x)
        # x = F.tanh(self.bn5(self.fc5(x)))
        # x = F.tanh(self.bn6(self.fc6(x)))
        x = self.fc10(x)
        return x.view(x.size(0), 64, 4, 8)

class TopModel(nn.Module):
    def __init__(self,gpu=False,input_size=4):
        super(TopModel,self).__init__()
        self.gpu = gpu
        self.linear = nn.Linear(64*input_size*8, 256)
        self.fc = nn.Linear(256, 256)
        self.output = nn.Linear(256, 10)

        self.apply(weights_init)

        if gpu:
            self.cuda()

    def forward(self,x):
        if self.gpu:
            x = x.cuda()
        B = x.size()[0]

        x = F.relu(self.linear(x.view(B,-1)))
        x = F.dropout(F.relu(self.fc(x)), 0.5, training=self.training)
        x = self.output(x)

        return x

class Model(nn.Module):
    def __init__(self, gpu=False,multies=2,unit = 0.5):
        super(Model, self).__init__()
        self.gpu = gpu
        self.multies = multies
        self.unit = unit
        self.other_unit = (1-unit)/(multies-1)
        self.models = nn.ModuleList([BottomModel(gpu) for i in range(self.multies)])
        self.predict = PredictModel(gpu)
        self.top = TopModel(gpu,int((int((int(32*self.unit)-2)/2+1)-2)/2+1)+(multies-2)*int((int((int(32*self.other_unit)-2)/2+1)-2)/2+1)+int((int((32-int(32*self.unit)-(multies-2)*int(32*self.other_unit)-2)/2+1)-2)/2+1))
        
        self.to(device)

    def forward(self, x, test_pred_perf=False):
        x = x.to(device)

        x_list = x.split([int(x.size()[2]*self.unit)]+[int(x.size()[2]*self.other_unit) for i in range(self.multies-2)]+[x.size()[2]-int(x.size()[2]*self.unit)-(self.multies-2)*int(x.size()[2]*self.other_unit)],dim=2)
        x_list = [self.models[i](x_list[i]) for i in range(self.multies)]
        # print(f"BottomModel output shape: {x_list[0].shape}")
        
        # Predict the output of the second bottom model
        # predicted_output_1 意思 是 prediction model 預測 bottom model 1 的輸出
        # 不回傳 gradient 至 bottom model 2
        predicted_output_1 = self.predict(x_list[1].detach()) 

        # 回傳 gradient 至 bottom model 2
        # predicted_output_1 = self.predict_model(x_list[1])
        if test_pred_perf:
            x = torch.cat((predicted_output_1,x_list[1]),dim=2)
            
        else:
            x = torch.cat(x_list,dim=2)
        x = self.top(x)
        return x, x_list[0] ,x_list[1], predicted_output_1
        # x:top model output vector
        # x_list[0]:passive party(client/attacker) embedding vector
        # x_list[1]:active party(host) embedding vector
        # predicted_output_1:prediction model output vector



    def loss(self, pred, label, output_1, predicted_output_1):
        label = label.to(device)
        # B = output_1.size()[0]

        # output_1_reshaped = output_1.view(B, -1)
        # predicted_output_1_reshaped = predicted_output_1.view(B, -1)

        # target = torch.ones(B).to(device)
        
        classification_loss = F.cross_entropy(pred, label)
        prediction_loss = F.mse_loss(predicted_output_1, output_1)
        # cosine_loss = nn.CosineEmbeddingLoss()
        # prediction_loss = cosine_loss(predicted_output_1_reshaped, output_1_reshaped, target)
        # prediction_loss = F.l1_loss(predicted_output_1, output_1)
        return classification_loss, prediction_loss