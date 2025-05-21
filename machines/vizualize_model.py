import torch
from torchviz import make_dot
from model import ANPRModel 


if __name__ == "__main__":

    model = ANPRModel()
    dummy_input = torch.randn(1, 3, 224, 224)
    output = model(dummy_input)
    dot = make_dot(output, params=dict(model.named_parameters()))

    dot.format = "png"
    dot.render("anpr_model_visualization")
    print("Model visualization saved as 'anpr_model_visualization.png'")
    dot.view()
    