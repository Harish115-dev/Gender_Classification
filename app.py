from flask import Flask, request, jsonify, render_template
from PIL import Image
import torch
from torchvision import transforms
from model import Gender

app = Flask(__name__)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = Gender()
model.load_state_dict(torch.load("gender_model.pth", map_location=device))
model.to(device)
model.eval()

classes = ["female", "male"]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files["image"]
    img = Image.open(file.stream).convert("RGB")
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


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)