import onnxruntime as ort
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

isDisplayPredictText = False # 予測結果を表示するかどうか

# モデルパスを指定
model_path = 'mynn.onnx'

# ONNXランタイムセッションを作成
session = ort.InferenceSession(model_path)

# 入力データの名前を取得 (モデルに依存します)
input_name = session.get_inputs()[0].name

# 出力データの名前を取得 (モデルに依存します)
output_name = session.get_outputs()[0].name


def predict(PoseData):
    
    # ONNXランタイムで推論

    PoseData = np.array(PoseData, dtype=np.float32)
    # PoseDataを1×54の配列に変換
    PoseData = np.array(PoseData).reshape(1, -1)

    outputs = session.run([output_name], {input_name: PoseData })

    if isDisplayPredictText:
        print( outputs[0])
    return outputs[0]
    