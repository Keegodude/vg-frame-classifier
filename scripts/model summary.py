import torch
from torchsummary import summary
from torchviz import make_dot

from cnn_Model import ScreenshotToneClassifier
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ScreenshotToneClassifier()
model.to(device)

# Summary
summary(model, input_size=(3, 224, 224))

# Visualization
dummy_input = torch.randn(1, 3, 224, 224).to(device)
output = model(dummy_input)

make_dot(output, params=dict(model.named_parameters())).render("model_graph", format="png")