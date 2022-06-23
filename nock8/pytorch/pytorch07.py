#モデルの保存と読み込み
import torch
import torch.onnx as onnx
import torchvision.models as models

#モデルの重みの保存と読み込み
model = models.vgg16(pretrained=True)
torch.save(model.state_dict(), 'model_weights.pth')

model = models.vgg16()
model.load_state_dict(torch.load('model_weights.pth'))
model.eval()

#モデルの形ごと保存・読み込む方法
torch.save(model, 'model.pth')
model = torch.load('model.pth')

#ONNX形式でのモデル出力：Exporting Model to ONNX
input_image = torch.zeros((1,3,224,224))
onnx.export(model, input_image, 'model.onnx')