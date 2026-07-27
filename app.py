from flask import Flask,render_template,request,jsonify
import torch
from torchvision import transforms
from PIL import Image

app=Flask(__name__)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Gender()
model.load_state_dict(torch.load("gender_model.pth", map_location=device))
model.to(device)
model.eval()


classes = ["female", "male"]

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])


@app.route("/predict", methods=["POST"])
def predict():
    file=request.files("image")
    img=Image.open(file.stream).convert("RGB")
    img_t = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img_t)
        probs = torch.softmax(output, dim=1)
        pred = torch.argmax(probs, 1).item()
        confidence = probs[0][pred].item()

    return jsonify({
        "prediction": classes[pred],
        "confidence": round(confidence * 100, 2)
    })

    if __name__ == "__main__":
        app.run(debug=True)
    
