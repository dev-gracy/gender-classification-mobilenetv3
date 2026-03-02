import torch
from torchvision import transforms
from PIL import Image

# load model once
model = torch.jit.load("model.pth", map_location="cpu")
model.eval()

tf = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485,0.456,0.406],
        [0.229,0.224,0.225]
    )
])

def predict(image_path):
    """
    Returns:
    label: int (0 = Male, 1 = Female)
    confidence: float (0–1)
    """

    img = Image.open(image_path).convert("RGB")
    x = tf(img).unsqueeze(0)

    with torch.no_grad():
        out = model(x)
        probs = torch.softmax(out, dim=1)
        conf, pred = torch.max(probs, dim=1)

    label = int(pred.item())
    confidence = float(conf.item())

    return label, confidence
