import torch as tc
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 主な定数値
N_IN = 54  # 入力データの列数
N_MID = 100  # 中間層のニューロン数
N_OUT = 1  # 出力層のニューロン数
EPOCH = 1000
N_BATCH = 1
LR = 0.01  # 学習率

DisplayGraph = False

# CSVファイルの読み込みと前処理
def load_data(file_paths):
    train_x_list = []
    train_t_list = []
    test_x_list = []
    test_t_list = []

    for path in file_paths:
        df = pd.read_csv(path)
        
        # 学習データとテストデータに分割
        train_df = df[:-10]  # 最後から10行を除くすべての行
        test_df = df[-10:]  # 最後の10行

        # 入力データと目標値に分割
        train_x_list.append(train_df.iloc[:, :N_IN].values)
        train_t_list.append(train_df.iloc[:, N_IN].values.reshape(-1, 1))
        test_x_list.append(test_df.iloc[:, :N_IN].values)
        test_t_list.append(test_df.iloc[:, N_IN].values.reshape(-1, 1))

    # リストを結合してテンソルに変換
    train_x = tc.tensor(np.concatenate(train_x_list), dtype=tc.float32)
    train_t = tc.tensor(np.concatenate(train_t_list), dtype=tc.float32)
    test_x = tc.tensor(np.concatenate(test_x_list), dtype=tc.float32)
    test_t = tc.tensor(np.concatenate(test_t_list), dtype=tc.float32)

    return train_x, train_t, test_x, test_t

# ニューラルネットワークの定義
class MyNN(nn.Module):
    def __init__(self):
        super(MyNN, self).__init__()
        self.fc1 = nn.Linear(N_IN, N_MID)
        self.fc2 = nn.Linear(N_MID, N_MID)
        self.fc3 = nn.Linear(N_MID, N_OUT)

    def forward(self, x):
        x = tc.relu(self.fc1(x))
        x = tc.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 訓練と評価の関数
def train_and_test(model, train_x, train_t, test_x, test_t):
    optimizer = optim.SGD(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    # 訓練
    for epoch in range(EPOCH):
        model.train()
        optimizer.zero_grad()
        output = model(train_x)
        loss = criterion(output, train_t)
        loss.backward()
        optimizer.step()

        if (epoch+1) % 100 == 0:
            print(f'Epoch {epoch+1}, Loss: {loss.item()}')

    # テスト
    model.eval()
    with tc.no_grad():
        predictions = model(test_x)
        test_loss = criterion(predictions, test_t)
    print(f'Test Loss: {test_loss.item()}')
    
    dummy_input = tc.rand(N_BATCH, N_IN)
    tc.onnx.export(model, dummy_input, "mynn.onnx",export_params=True,input_names=['input'], output_names=['output'])


    print('Training Done and Model Exported as mynn.onnx')

    if DisplayGraph:
    # 実際の値と予測値のプロット
        plt.figure(figsize=(10, 4))
        plt.plot(test_t.numpy(), label='Actual', alpha=0.6)
        plt.plot(predictions.numpy(), label='Predicted', alpha=0.6)
        plt.legend()
        plt.show()
    
    

def MakeModelMainThread():
    file_paths = ['landmarks.csv']  # CSVファイルのパス
    train_x, train_t, test_x, test_t = load_data(file_paths)

    model = MyNN()
    train_and_test(model, train_x, train_t, test_x, test_t)
