import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def set_seed(seed=500):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU
set_seed(42)  # Call the seed function with your desired seed

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        # nn.init.xavier_uniform_(m.weight)
        nn.init.kaiming_uniform_(m.weight,nonlinearity='relu')
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

        '''
        測試維度對防禦效果之影響
        #4096
        '''
        self.batchnorm = nn.BatchNorm1d(2048)
        self.fully = nn.Linear(2048,10)
        
        self.apply(weights_init)

    def forward(self,x):
        x = x.to(device)

        x = F.relu(self.conv1(x)) 
        x = self.max_pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x)) 
        x = self.max_pool(F.relu(self.conv4(x))) 

        '''
        測試維度對防禦效果之影響
        #4096
        '''
        x = x.view(x.shape[0],-1)
        x = self.batchnorm(x)
        x = self.fully(F.relu(x))

        return x

# class PredictModel(nn.Module):
#     def __init__(self, gpu=False):
#         super(PredictModel, self).__init__()
#         input_dim = 64*4*8
#         self.gpu = gpu
#         self.fc1 = nn.Linear(input_dim, int(input_dim/64*63))  # 2048 -> 2016
#         self.bn1 = nn.BatchNorm1d(int(input_dim/64*63))
#         self.fc2 = nn.Linear(int(input_dim/64*63), int(input_dim/32*31))  # 2016 -> 1984
#         self.bn2 = nn.BatchNorm1d(int(input_dim/32*31))
#         self.fc3 = nn.Linear(int(input_dim/32*31), int(input_dim/16*15))  # 1984 -> 1920
#         self.bn3 = nn.BatchNorm1d(int(input_dim/16*15))
#         self.fc4 = nn.Linear(int(input_dim/16*15), int(input_dim/8*7))    # 1920 -> 1792
#         self.bn4 = nn.BatchNorm1d(int(input_dim/8*7))
#         self.fc5 = nn.Linear(int(input_dim/8*7), int(input_dim/4*3))      # 1792 -> 1536
#         self.bn5 = nn.BatchNorm1d(int(input_dim/4*3))
#         self.fc6 = nn.Linear(int(input_dim/4*3), int(input_dim/8*7))      # 1536 -> 1792
#         self.bn6 = nn.BatchNorm1d(int(input_dim/8*7))
#         self.fc7 = nn.Linear(int(input_dim/8*7), int(input_dim/16*15))    # 1792 -> 1920
#         self.bn7 = nn.BatchNorm1d(int(input_dim/16*15))
#         self.fc8 = nn.Linear(int(input_dim/16*15), int(input_dim/32*31))  # 1920 -> 1984
#         self.bn8 = nn.BatchNorm1d(int(input_dim/32*31))
#         self.fc9 = nn.Linear(int(input_dim/32*31), int(input_dim/64*63))  # 1984 -> 2016
#         self.bn9 = nn.BatchNorm1d(int(input_dim/64*63))
#         self.fc10 = nn.Linear(int(input_dim/64*63), input_dim)  
#         self.dropout = nn.Dropout(0.5)

#         self.apply(weights_init)

#     def forward(self, x):
#         x = x.to(device)

#         x = x.view(x.size(0), -1)
#         # print(f"PredictModel input shape: {x.shape}")
#         ## 調整成 tanh 看看
#         x = F.leaky_relu(self.bn1(self.fc1(x)))
#         x = F.leaky_relu(self.bn2(self.fc2(x)))
#         x = F.leaky_relu(self.bn3(self.fc3(x)))
#         x = F.leaky_relu(self.bn4(self.fc4(x)))
#         # x = F.tanh(self.fc4(x))
#         x = F.leaky_relu(self.bn5(self.fc5(x)))
#         # x = F.tanh(self.fc5(x))
#         x = F.leaky_relu(self.bn6(self.fc6(x)))
#         # x = F.tanh(self.fc6(x))
#         x = F.leaky_relu(self.bn7(self.fc7(x)))
#         x = F.leaky_relu(self.bn8(self.fc8(x)))
#         x = F.leaky_relu(self.bn9(self.fc9(x)))
#         # x = F.tanh(self.fc7(x))
#         # x = F.tanh(self.fc8(x))
#         # x = F.tanh(self.fc9(x))
#         # x = F.tanh(self.bn1(self.fc1(x)))
#         # x = self.dropout(x)
#         # x = F.tanh(self.bn2(self.fc2(x)))
#         # # x = self.dropout(x)
#         # x = F.tanh(self.bn3(self.fc3(x)))
#         # # x = self.dropout(x)
#         # x = F.tanh(self.bn4(self.fc4(x)))
#         # # x = self.dropout(x)
#         # x = F.tanh(self.bn5(self.fc5(x)))
#         # x = F.tanh(self.bn6(self.fc6(x)))
#         x = self.fc10(x)
#         return x.view(x.size(0), 64, 4, 8)

class PredictModel(nn.Module):
    def __init__(self, gpu=False, multies=2):
        super(PredictModel, self).__init__()
        '''
        測試維度對防禦效果之影響
        #4096
        self.input_dim = 64 * 8 * 8  
        input_dim = 64 * 8 * 8 # 4096
        '''
        
        self.input_dim = 20
        input_dim = 20
        self.num_participants = multies
        self.bottom_output_dim = input_dim // multies
        # x = x.view(x.size(0), -1)

        # Encoder part (原始的 fc1 到 fc3)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, int(input_dim/64*63)),  # 4096 -> 4032
            nn.BatchNorm1d(int(input_dim/64*63)),
            nn.ReLU(True),
            
            nn.Linear(int(input_dim/64*63), int(input_dim/32*31)),  # 4032 -> 3906
            nn.BatchNorm1d(int(input_dim/32*31)),
            nn.ReLU(True),
            
            nn.Linear(int(input_dim/32*31), int(input_dim/16*15)),  # 3906 -> 3662
            nn.BatchNorm1d(int(input_dim/16*15)),
            nn.ReLU(True),

        #     nn.Linear(int(input_dim/16*15), int(input_dim/8*7)),  # 3662 -> 3204
        #     nn.BatchNorm1d(int(input_dim/8*7)),
        #     nn.ReLU(True),

        #     nn.Linear(int(input_dim/8*7), int(input_dim/4*3)),  # 3204 -> 2403
        #     nn.BatchNorm1d(int(input_dim/4*3)),
        #     nn.ReLU(True),
        )
        
        # Decoder part (原始的 fc6 到 fc10)
        self.decoder = nn.Sequential(
            # nn.Linear(int(input_dim/4*3), int(input_dim/8*7)),  # 2403 -> 3204
            # nn.BatchNorm1d(int(input_dim/8*7)),
            # nn.ReLU(True),

            # nn.Linear(int(input_dim/8*7), int(input_dim/16*15)),  # 3204 -> 3662
            # nn.BatchNorm1d(int(input_dim/16*15)),
            # nn.ReLU(True),

            nn.Linear(int(input_dim/16*15), int(input_dim/32*31)),  # 3662 -> 3906
            nn.BatchNorm1d(int(input_dim/32*31)),
            nn.ReLU(True),
            
            nn.Linear(int(input_dim/32*31), int(input_dim/64*63)),  # 3906 -> 4032
            nn.BatchNorm1d(int(input_dim/64*63)),
            nn.ReLU(True),
            
            nn.Linear(int(input_dim/64*63), input_dim),              # 4032 -> 4096
            nn.BatchNorm1d(int(input_dim))
        )
        
        self.dropout = nn.Dropout(0.1)
        self.train_mode = True
        self.s_score_threshold = 2.0
        
        # Model properties
        self.normalize = True
        self.unit = False
        # self.unit = True
        
        # Normalization parameters
        self.mu = 0
        self.std = 1
        
        # 初始化權重
        self.apply(weights_init)
        
        # 設置設備
        self.device = torch.device("cuda" if torch.cuda.is_available() and gpu else "cpu")
        self.to(self.device)
        
    def forward(self, x):
        input_tensor = x.to(self.device)
        # print("MAE input tensor size:", input_tensor.shape)
        
        if self.train_mode:
            # Training mode
            if self.normalize:
                input_tensor = (input_tensor - self.mu) / self.std
                orig_input_tensor = input_tensor.clone()
                input_tensor = input_tensor.view(input_tensor.shape[0],-1)
            # elif self.unit:
            #     orig_input_tensor = input_tensor.clone()
            #     orig_norm = torch.zeros([orig_input_tensor.shape[0], self.num_participants]).to(self.device)
            #     for i in range(self.num_participants):
            #         orig_norm[:,i] = torch.norm(input_tensor[:, :,i*self.bottom_output_dim:(i+1)*self.bottom_output_dim, :], dim=2)
            #         input_tensor[:, :, i*self.bottom_output_dim:(i+1)*self.bottom_output_dim, :] = F.normalize(
            #             input_tensor[ :, :, i*self.bottom_output_dim:(i+1)*self.bottom_output_dim, :], dim=2
            #         )
            
            # latent = self.encoder(input_tensor)
            # output = self.decoder(latent)
            
            '''
            需要 output 2個結果，分別為 N-1 以及 1-1 的 strategy，以利後面計算 loss
            步驟:
            1. 創建 mask
            2. 讓 input 乘上 mask
            3. 丟入 MAE
            '''
            """
            前向傳播邏輯
            
            Args:
                x (torch.Tensor): 輸入張量，包含所有參與者的特徵
            
            Returns:
                dict: 包含 N-1 和 1-1 策略的重建結果
            """
            
            # N-1 策略
            masked_input_n_minus_one, mask_n_minus_one, target_mask_n_minus_one = self.create_mask(x, mode='n_minus_one')
            latent_n_minus_one = self.encoder(masked_input_n_minus_one.view(x.size(0), -1))
            output_n_minus_one = self.decoder(latent_n_minus_one)
            
            # Safely modify output based on target mask
            if target_mask_n_minus_one is not None:
                # Only modify the parts of the output corresponding to the masked participant
                non_zero_indices = (target_mask_n_minus_one.view(-1) == 1).nonzero(as_tuple=True)[0]
                if len(non_zero_indices) > 0:
                    output_n_minus_one_masked = output_n_minus_one.clone()
                    output_n_minus_one_masked.view(-1)[non_zero_indices] = 0
                else:
                    output_n_minus_one_masked = output_n_minus_one
            else:
                output_n_minus_one_masked = output_n_minus_one
            
            # 1-1 策略
            masked_input_one_to_one, mask_one_to_one, target_mask_one_to_one = self.create_mask(x, mode='one_to_one')
            latent_one_to_one = self.encoder(masked_input_one_to_one.view(x.size(0), -1))
            output_one_to_one = self.decoder(latent_one_to_one)
            
            # Safely modify output for one-to-one strategy
            if target_mask_one_to_one is not None:
                non_zero_indices = (target_mask_one_to_one.view(-1) == 1).nonzero(as_tuple=True)[0]
                if len(non_zero_indices) > 0:
                    output_one_to_one_masked = output_one_to_one.clone()
                    output_one_to_one_masked.view(-1)[non_zero_indices] = 0
                else:
                    output_one_to_one_masked = output_one_to_one
            else:
                output_one_to_one_masked = output_one_to_one
            
            # 原始輸入的編碼和解碼
            input_tensor = self.dropout(input_tensor)
            latent = self.encoder(input_tensor)
            output = self.decoder(latent)
            
            
            if self.normalize:
                output = (output * self.std) + self.mu
            # elif self.unit:
            #     for i in range(self.num_participants):
            #         output[:, i*self.bottom_output_dim:(i+1)*self.bottom_output_dim] = (
            #             output[:, i*self.bottom_output_dim:(i+1)*self.bottom_output_dim] * 
            #             orig_norm[:,i].unsqueeze(1)
            #         )
            
            return {
                'output_n_minus_one': output_n_minus_one,
                'mask_n_minus_one': mask_n_minus_one,
                'target_mask_n_minus_one': target_mask_n_minus_one,
                'output_one_to_one': output_one_to_one,
                'mask_one_to_one': mask_one_to_one,
                'target_mask_one_to_one': target_mask_one_to_one,
                'original_input': x,
                'output': output,
            }
        
        else:
            # Evaluation mode
            print("Eval mode")
            if self.normalize:
                input_tensor = (input_tensor - self.mu) / self.std #標準化
                orig_input_tensor = input_tensor.clone() 
                input_tensor = input_tensor.view(input_tensor.shape[0],-1)
            # elif self.unit:
            #     orig_input_tensor = input_tensor.clone()
            #     orig_norm = torch.zeros([orig_input_tensor.shape[0], self.num_participants]).to(self.device)
            #     for i in range(self.num_participants):
            #         orig_norm[:,i] = torch.norm(input_tensor[:, :, i*self.bottom_output_dim:(i+1)*self.bottom_output_dim, :], dim=2)
            #         input_tensor[:, :, i*self.bottom_output_dim:(i+1)*self.bottom_output_dim, :] = F.normalize(
            #             input_tensor[:, :, i*self.bottom_output_dim:(i+1)*self.bottom_output_dim, :], dim=2
            #         )
            #     orig_input_tensor = input_tensor.clone()
            
            # Calculate reconstruction scores for each party
            s_score = torch.zeros([input_tensor.shape[0], self.num_participants, self.num_participants]).to(self.device) #建立每個 sample 每個參與者對每個參與者的 s_score 計算 
            print("s_score size:", s_score.shape)
            for i in range(self.num_participants):
                mask = torch.zeros_like(orig_input_tensor).to(self.device) #建立遮住某個參與者輸入的遮罩
                '''
                測試維度影響防禦效能
                #4096
                mask[:, :, i*self.bottom_output_dim:(i+1)*self.bottom_output_dim, :] = 1
                '''
                mask[:, i*self.bottom_output_dim:(i+1)*self.bottom_output_dim] = 1
                

                # print("mask shape", mask.shape)
                # print("input shape:", input_tensor.shape)
                mask_input = (mask * orig_input_tensor).view(input_tensor.shape)
                # print("mask_input shape:", mask_input.shape)
                latent = self.encoder(mask_input)
                recon = self.decoder(latent)
                # recon_dim_recover = recon.view(input_tensor.shape)
                
                for j in range(self.num_participants):
                    if i == j:
                        continue
                    norm_for_j = torch.norm(
                        recon[ :, j*self.bottom_output_dim:(j+1)*self.bottom_output_dim] - 
                        input_tensor[ :, j*self.bottom_output_dim:(j+1)*self.bottom_output_dim],
                        dim=1
                    ) 
                    for k in range(norm_for_j.shape[0]):  # 計算遮住每個參與者輸入與其他參與者預期的差距，存在 s_score list 裡面 
                        s_score[:,i,j] = norm_for_j[k]

            # Create mask based on voting
            voting_matrix = torch.sum((s_score > self.s_score_threshold).int(), dim=1) # 根據每筆 sample 統計哪幾個參與者在該 sample 大於閾值應該被替換
            MAE_mask = torch.ones_like(input_tensor)
            # print("mask dim:",mask.shape)
            
            '''
            測試 mask 到底有沒有用
            '''
            count = 0

            voting_matrix = voting_matrix.cpu().numpy()
            f = open("voting_record.txt","a")
            np.savetxt(f,voting_matrix)
            f.close()
            
            # print(voting_matrix.shape)

            for i in range(voting_matrix.shape[0]): # i 代表 sample number
                for j in range(voting_matrix.shape[1]): # j 代表在該 sample number 中每個參與者被投票的次數
                    # print(voting_matrix[i,j].item())
                    if voting_matrix[i,j].item() >= int(self.num_participants/2):
                        # print(voting_matrix[i,j].item())
                        count+=1
                        
                        '''
                        確認維度
                        '''
                        # mask[i, :, j*self.bottom_output_dim:(j+1)*self.bottom_output_dim, :] = 0 # 建立遮罩，屏蔽該參與者在該 sample 輸入，讓 MAE 緩解
                        MAE_mask[i, j*self.bottom_output_dim:(j+1)*self.bottom_output_dim] = 0 # 建立遮罩，屏蔽該參與者在該 sample 輸入，讓 MAE 緩解
                       

            print("count 為 :",count)
            

            
            # Final reconstruction
            # print("input tensor size:",input_tensor.shape)
            # print("mask size:", mask.shape)
            latent = self.encoder(input_tensor * MAE_mask)
            output = self.decoder(latent)

            # j = open("mask_record.txt","a")
            # mask_record = mask.cpu().numpy()
            # np.savetxt(j,mask_record)
            # j.close()
            
            if self.normalize:
                output = (output * self.std) + self.mu
            # elif self.unit:
            #     for i in range(self.num_participants):
            #         output[:, i*self.bottom_output_dim:(i+1)*self.bottom_output_dim] = (
            #             output[:, i*self.bottom_output_dim:(i+1)*self.bottom_output_dim] * 
            #             orig_norm[:,i].unsqueeze(1)
            #         )
        
            return output
    
    def create_mask(self, input_tensor, mode='n_minus_one'):
        """
        創建遮罩
        
        Args:
            input_tensor (torch.Tensor): 輸入張量
            mode (str): 遮罩模式 - 'n_minus_one' 或 'one_to_one'
        
        Returns:
            torch.Tensor: 遮罩後的輸入張量
        """
        
        original_shape = input_tensor.shape
        batch_size = input_tensor.size(0)
        #這邊分割方法要視每個模型設定而定
        '''
        測試維度影響防禦效能
        #4096
        Embedding_vector_dim_size = input_tensor.size(2)
        '''
        Embedding_vector_dim_size = input_tensor.size(1)
        
        Embedding_vector_dim_per_participant = Embedding_vector_dim_size // self.num_participants

        # 初始化遮罩和目標遮罩
        target_mask = torch.zeros_like(input_tensor)

        if mode == 'n_minus_one':
            # 隨機選擇一個參與者
            mask = torch.ones_like(input_tensor)
            masked_participant = torch.randint(0, self.num_participants, (1,)).item()
            for b in range(batch_size):
                start_idx = masked_participant * Embedding_vector_dim_per_participant
                end_idx = (masked_participant + 1) * Embedding_vector_dim_per_participant
                # print("n-1 to 1 mask 維度:",start_idx,"-",end_idx-1)
                '''
                測試維度影響防禦效能
                #4096
                mask[b, :, start_idx:end_idx,:] = 0  # 遮住該參與者的特徵
                target_mask[b, :, start_idx:end_idx,:] = 1  # 設置目標遮罩為該參與者的特徵
                '''
                mask[b, start_idx:end_idx] = 0  
                target_mask[b, start_idx:end_idx] = 1
                

        elif mode == 'one_to_one':
            # 隨機選擇兩個不同的參與者
            mask = torch.zeros_like(input_tensor)
            participant_pairs = torch.randperm(self.num_participants)[:2]
            for b in range(batch_size):
                # 保留第一個參與者
                start_idx_1 = participant_pairs[0] * Embedding_vector_dim_per_participant
                end_idx_1 = (participant_pairs[0] + 1) * Embedding_vector_dim_per_participant
                '''
                測試維度影響防禦效能
                #4096
                mask[b, :, start_idx_1:end_idx_1, :] = 1
                '''
                mask[b, start_idx_1:end_idx_1] = 1

                # print("1 to 1 保留者1 mask 維度:",start_idx_1,"-",end_idx_1-1)

                # 設置目標遮罩為第二個參與者
                start_idx_2 = participant_pairs[1] * Embedding_vector_dim_per_participant
                end_idx_2 = (participant_pairs[1] + 1) * Embedding_vector_dim_per_participant
                '''
                測試維度影響防禦效能
                #4096
                target_mask[b, :, start_idx_2:end_idx_2, :] = 1
                '''
                target_mask[b, start_idx_2:end_idx_2] = 1

                # print("1 to 1 保留者2 mask 維度:",start_idx_2,"-",end_idx_2-1)

        else:
            raise ValueError("Unsupported mode. Use 'n_minus_one' or 'one_to_one'.")

        # 確保遮罩形狀一致
        mask = mask.view(original_shape)
        target_mask = target_mask.view(original_shape)

        return input_tensor * mask, mask, target_mask
        
        # original_shape = input_tensor.shape
        # print("original_shape:",original_shape)
        
        # batch_size = input_tensor.size(0)
        # target_mask = torch.zeros_like(input_tensor)
        
        # if mode == 'n_minus_one':
        #     mask = torch.ones_like(input_tensor)
        #     # 隨機選擇要遮罩的參與者
        #     masked_participant = torch.randint(self.num_participants,(1,)).item()
        #     for b in range(batch_size):
        #         mask[b, masked_participant*self.input_dim : (masked_participant+1)*self.input_dim] = 0
        #         # print("Mask size:",mask.size())
                
        #         target_mask[b,masked_participant*self.input_dim : (masked_participant+1)*self.input_dim] = 1
        #         # print("Target Mask size:",mask.size())
        
        # elif mode == 'one_to_one':
        #     mask = torch.zeros_like(input_tensor)
        #     # 隨機選擇兩個不同的參與者
        #     participant_pairs = torch.randperm(self.num_participants)[:2]
        #     for b in range(batch_size):
        #         mask[b, participant_pairs[0]*self.input_dim : (participant_pairs[0]+1)*self.input_dim] = 1
        #         target_mask[b,participant_pairs[1]*self.input_dim : (participant_pairs[1]+1)*self.input_dim] = 1
        
        # # 重新調整回原始形狀
        # mask = mask.view(original_shape)
        # target_mask = target_mask.view(original_shape)
        
        # print("mask sum:",mask.sum())
        # print("target mask sum",target_mask.sum())
        
        # return input_tensor * mask, mask, target_mask
        
    
    def set_mode(self, train=True):
        """設置模型模式"""
        self.train_mode = train

class VerticalFLMAELoss(nn.Module):
    def __init__(self, alpha=1, beta=0.1):
        """
        垂直聯邦學習 MAE 的 Loss 函數s
        
        Args:
            alpha (float): N-1 策略的權重
            beta (float): 1-1 策略的權重
        """
        super(VerticalFLMAELoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
    
    def forward(self, model_output):
        """
        計算 Loss
        
        Args:
            model_output (dict): MAE 模型的輸出
        
        Returns:
            torch.Tensor: 總 Loss
        """
        output_n_minus_one = model_output['output_n_minus_one']
        mask_n_minus_one = model_output['mask_n_minus_one']
        target_mask_n_minus_one = model_output['target_mask_n_minus_one']
        output_one_to_one = model_output['output_one_to_one']
        mask_one_to_one = model_output['mask_one_to_one']
        target_mask_one_to_one = model_output['target_mask_one_to_one']
        original_input = model_output['original_input']
        
        # print("輸入檢查:")
        # for tensor_name, tensor in model_output.items():
        #     if torch.isnan(tensor).any():
        #         print(f"{tensor_name} contains NaN values.")
        #     # else:
        #     #     print(f"{tensor_name} does not contain NaN values.")
        
        # for tensor_name, tensor in model_output.items():
        #     if torch.isinf(tensor).any():
        #         print(f"{tensor_name} contains NaN values.")
        
        # print("遮罩檢查:")
        # print("mask_n_minus_one sum:", mask_n_minus_one.sum())
        # print("target_mask_n_minus_one sum:", target_mask_n_minus_one.sum())
        # print("mask_one_to_one sum:", mask_one_to_one.sum())
        # print("target_mask_one_to_one sum:", target_mask_one_to_one.sum())
        
        # print("數值穩定性檢查:")
        # print("output_n_minus_one range:", output_n_minus_one.min(), output_n_minus_one.max())
        # print("original_input range:", original_input.min(), original_input.max())
        
        # print("防止零除問題:")
        # n_minus_one_valid = (mask_n_minus_one == 0).sum().item()
        # one_to_one_valid = (mask_one_to_one == 0).sum().item()

        # if n_minus_one_valid == 0 or one_to_one_valid == 0:
        #     print("Warning: No valid elements for loss calculation.")
        #     return torch.tensor(0.0, requires_grad=True).to(original_input.device)
        
        B = mask_n_minus_one.size()[0]
        mask_n_minus_one = mask_n_minus_one.view(B,-1)
        target_mask_n_minus_one = target_mask_n_minus_one.view(B,-1)
        
        mask_one_to_one = mask_one_to_one.view(B,-1)
        target_mask_one_to_one = target_mask_one_to_one.view(B,-1)
        
        original_input = original_input.view(B,-1)
        
        
        
        # N-1 策略 Loss
        loss_n_minus_one = F.mse_loss(
            output_n_minus_one[mask_n_minus_one == 0], 
            original_input[target_mask_n_minus_one == 1]
        )
        
        # 1-1 策略 Loss
        loss_one_to_one = F.mse_loss(
            output_one_to_one[target_mask_one_to_one == 1], 
            original_input[target_mask_one_to_one == 1]
        )
        
        # 總 Loss
        # print("loss_n_minus_one:",loss_n_minus_one)
        # print("loss_one_to_one:",loss_one_to_one)
        # if torch.isnan(loss_n_minus_one).any():
        #     loss_n_minus_one = 0.0001
        # if torch.isnan(loss_one_to_one).any():
        #     loss_one_to_one = 0.0001
        
        # print("loss_n_minus_one:",loss_n_minus_one)
        # print("loss_one_to_one:",loss_one_to_one)
        total_loss = self.alpha * loss_n_minus_one + self.beta * loss_one_to_one
        
        return total_loss
        
class TopModel(nn.Module):
    def __init__(self,gpu=False,input_size=4):
        super(TopModel,self).__init__()
        self.gpu = gpu
        '''
        測試維度對防禦之影響
        #4096
        self.linear = nn.Linear(64*input_size*8, 256)
        self.fc = nn.Linear(256, 256)
        self.output = nn.Linear(256, 10)
        '''
        self.linear = nn.Linear(20, 256)
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
        self.predict = PredictModel(gpu,multies)
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
        '''
        測試維度影響防禦效能
        #4096
        Embedding_Vector = torch.cat(x_list,dim=2)
        '''
        Embedding_Vector = torch.cat(x_list,dim=1)
        main_pred = self.top(Embedding_Vector)
        
        if test_pred_perf: # 測試 MAE 效能 
            self.predict.set_mode(False) #不訓練
            MAE_output = self.predict(Embedding_Vector.detach()) 
            MAE_pred = self.top(MAE_output.detach())
            
        else: # 單純訓練 查看 model 本身準確度
            self.predict.set_mode(True) #訓練
            '''
            測試維度影響防禦效能
            #4096
            x = torch.cat(x_list,dim=2)
            '''
            x = torch.cat(x_list,dim=1)
            MAE_output = self.predict(Embedding_Vector.detach())
            MAE_pred = self.top(MAE_output['output'].detach())
        
        return main_pred, Embedding_Vector, MAE_output, MAE_pred
        # x:top model output vector
        # x_list[0]:passive party(client/attacker) embedding vector
        # x_list[1]:active party(host) embedding vector
        # predicted_output_1:prediction model output vector



    def loss(self, pred, label, Bottom_output, MAE_output):
        label = label.to(device)
        # B = output_1.size()[0]

        # output_1_reshaped = output_1.view(B, -1)
        # predicted_output_1_reshaped = predicted_output_1.view(B, -1)

        # target = torch.ones(B).to(device)
        
        classification_loss = F.cross_entropy(pred, label)
        
        '''
        查看每個 sample 的 loss 值
        # classification_loss_each = F.cross_entropy(pred, label, reduction='none')
        # return classification_loss, prediction_loss, classification_loss_each
        '''
        
        ''' 需要撰寫 VFLIP 的 loss'''
        MAE_loss_class = VerticalFLMAELoss()
        MAE_loss = MAE_loss_class(MAE_output)
        # prediction_loss = F.l1_loss(predicted_output_1, output_1)
        return classification_loss, MAE_loss