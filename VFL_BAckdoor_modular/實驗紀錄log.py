'''
目前問題:
1. 2個參與者時，攻擊者會被偵測到9成9，另一參與者會被偵測到2成4
    要嘗試減少另一參與者被偵測到的比例，否則會有一堆不可用數據 (可從 shapley value 分布下手)
2. 在乾淨的測試集經過SHAP淨化後準確率太低，初步推測是還原的能力太差，可嘗試加深度
'''

'''2/12
改為 查看測試集每個人在該預測類別時的 shapley value，若在訓練時所搜集的分佈外，則列為受感染樣本，
針對正的 shapley value 視為後門案例，將其遮住並緩解
需要做到以下功能：
1. 訓練時搜集的圖上畫出平均數以及中位數的位置，並測試在幾個標準差可以容納大部分樣本
2. 測試下毒測試集樣本在預測的 label 時，攻擊者以及正常參與者的分布會在什麼地方
3. 你很棒
'''

'''2/14
有一篇論文在做水平的後門偵測，shapley-base，偵測時會使用到 MAD(Median Absolute Deviation)
偵測方法細節有需要可以參考：
BatFL:Backdoordetectiononfederated learning in e-Health.
https://ieeexplore.ieee.org/abstract/document/9521339
'''

'''3/19
使用具類方法效果不如原先 prediction model
可以嘗試將距離計算方法改為 Cosine
如果效果還是不佳，方法改為針對每個參與者都訓練一個 active party 對標到 passive_party 的 prediction model
'''

'''3/21
Cosine 效果與 prediction model 接近
'''

'''3/27
修好了 VFLIP 的緩解功能，要注意 voting 以及 sscore 的 dim ，設錯直接爆炸
'''

'''4/7
SHAP Prediction model 能夠使用，接下來嘗試 MAE 是否可以較有效改善 ACC 以及 ASR
紀錄 VFLIP 在 8 人以及 4 人的 TIFS 攻擊下較佳的 threshold
目前發現 SHAP 在越來越多人的情況下會不容易辨識錯誤者，因為 shapley value 會隨著人數上升而被稀釋
目前想法是可以根據人數的提升減少 threshold 與 Q2 的距離，例如可以計算四分位距，並除以人數

另外有一篇論文可能可以拿來改 MAE 框架:
https://openaccess.thecvf.com/content/CVPR2022/html/He_Masked_Autoencoders_Are_Scalable_Vision_Learners_CVPR_2022_paper
eval.py 這裡改
'''

'''4/9
SHAP_MAE 撰寫完成，仍需 debug，似乎無法捕捉到要緩解的模型
目前實驗結果說明 SHAP 方法在人多的時候 ACC 與 VFLIP 差不多，但 ASR 會差距3-4%

後續須做動作:
1. 論文撰寫:動機以及貢獻部分要著重描寫，另外可以考慮加上 System overview，增加字數也能夠增加繪圖量
2. 論文繪圖:整體架構圖以及步驟圖
3. SHAP_MAE debug
4. 嘗試 CVPR2022 那篇論文的方法嘗試提高 SHAP_MAE ACC
'''

'''4/12

trace: 
043% 1. 論文撰寫:動機以及貢獻部分要著重描寫，另外可以考慮加上 System overview，增加字數也能夠增加繪圖量 -> 論文填寫到 Related work
000% 2. 論文繪圖:整體架構圖以及步驟圖
100% 3. SHAP_MAE debug -> 完成
100% 4. 嘗試 CVPR2022 那篇論文的方法嘗試提高 SHAP_MAE ACC -> 目前看來沒什麼用，但主要問題是 threshold 設定不精確，需要找方法突破

嘗試將訓練時蒐集的 shap value 每一筆都 歸一化
'''

'''6/9
新論文：https://ieeexplore.ieee.org/abstract/document/10988592
'''

'''7/28
BM 資料及兩人實驗 batch-size 設為128, 將主要特徵分配給攻擊者以及主動方效果最好
目前結果:
===========================================
2人:
現在正在計算測試時的 shapley value，進度為：42.0%                                   
poisoned 模型 acc on 乾淨測試資料集: 0.8446
乾淨測試的 shapley 計算完畢，正在繪圖...
正在測試乾淨測試集......
共有 242 筆資料被偵測到
經過 SHAP_MAE 淨化後 poisoned 模型 acc on 乾淨測試資料集: 0.8312
====================
現在正在計算測試時的 shapley value，進度為：89.0%                     未經防禦的情況下:
ACC 為 0.5432
ASR 為 0.9794
排除掉原先 label 即為 0 之資料後，ASR 為 0.9603
--------------------
正在測試下毒測試集......
SHAP(MAE) 防禦效果:
共有 2180 筆資料被偵測到，其中有 0.0 筆資料本來 label 就是0
acc on 下毒資料集:0.8321
ASR on 下毒資料集: 0.4756
排除掉原先 label 即為 0 之資料後，ASR 為 0.1238
===========================================

4人:
現在正在計算測試時的 shapley value，進度為：42.0%                                   
poisoned 模型 acc on 乾淨測試資料集: 0.8240
乾淨測試的 shapley 計算完畢，正在繪圖...
正在測試乾淨測試集......
共有 699 筆資料被偵測到
經過 SHAP_MAE 淨化後 poisoned 模型 acc on 乾淨測試資料集: 0.7506
====================
現在正在計算測試時的 shapley value，進度為：89.0%                     未經防禦的情況下:
ACC 為 0.5410
ASR 為 0.9790
排除掉原先 label 即為 0 之資料後，ASR 為 0.9622
--------------------
正在測試下毒測試集......
SHAP(MAE) 防禦效果:
共有 1392 筆資料被偵測到，其中有 0.0 筆資料本來 label 就是0
acc on 下毒資料集:0.7855
ASR on 下毒資料集: 0.5114
排除掉原先 label 即為 0 之資料後，ASR 為 0.2108
===========================================

8人:
現在正在計算測試時的 shapley value，進度為：42.0%                                   
poisoned 模型 acc on 乾淨測試資料集: 0.8213
乾淨測試的 shapley 計算完畢，正在繪圖...
正在測試乾淨測試集......
共有 705 筆資料被偵測到
經過 SHAP_MAE 淨化後 poisoned 模型 acc on 乾淨測試資料集: 0.7371
====================
現在正在計算測試時的 shapley value，進度為：89.0%                     未經防禦的情況下:
ACC 為 0.5562
ASR 為 0.9592
排除掉原先 label 即為 0 之資料後，ASR 為 0.9253
--------------------
正在測試下毒測試集......
SHAP(MAE) 防禦效果:
共有 1259 筆資料被偵測到，其中有 0.0 筆資料本來 label 就是0
acc on 下毒資料集:0.7721
ASR on 下毒資料集: 0.5069
排除掉原先 label 即為 0 之資料後，ASR 為 0.2202
'''