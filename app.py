import os
import shutil
import uuid
from pathlib import Path

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from flask import Flask, request, jsonify
from PIL import Image
from torchvision import models, transforms
import shap

app = Flask(__name__)

MODEL_PATH = 'model/best_model.pth'
CLASSES_PATH = 'model/classes.txt'

BASE_TEMP_DIR = Path('temp')
BASE_TEMP_DIR.mkdir(exist_ok=True)
STATIC_IMG_DIR = Path('static')
STATIC_IMG_DIR.mkdir(exist_ok=True)


def get_session_temp_dirs():
    session_id = str(uuid.uuid4())[:8]
    temp_dir = BASE_TEMP_DIR / session_id
    input_dir = temp_dir / 'input'
    output_dir = temp_dir / 'output'
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    return temp_dir, input_dir, output_dir


def save_uploaded_file(uploaded_file, save_dir: Path) -> Path:
    file_path = save_dir / uploaded_file.filename
    uploaded_file.save(file_path)
    return file_path


# ==================== Base Model Runner ====================

class BaseModelRunner:
    def __init__(self, model_path, classes_path, device=None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = models.resnet50(weights=None)
        num_classes = sum(1 for _ in open(classes_path))
        self.model.fc = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(self.model.fc.in_features, num_classes)
        )
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device).eval()

        with open(classes_path) as f:
            self.classes = [l.strip() for l in f]

        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])

    def _process_image(self, img_path):
        img = Image.open(img_path).convert('RGB')
        inp = self.transform(img).unsqueeze(0).to(self.device)
        return img, inp

    def _generate_output_image(self, original_img, explanation_map, output_path, title, alpha=0.5):
        explanation_resized = cv2.resize(explanation_map, original_img.size, interpolation=cv2.INTER_LINEAR)
        heatmap = cv2.applyColorMap((explanation_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(
            cv2.cvtColor(np.array(original_img), cv2.COLOR_RGB2BGR),
            1 - alpha, heatmap, alpha, 0
        )

        fig, ax = plt.subplots(1, 3, figsize=(18, 6))
        ax[0].imshow(original_img)
        ax[0].axis('off')
        ax[0].set_title('Original')
        im = ax[1].imshow(explanation_resized, cmap='jet')
        ax[1].axis('off')
        ax[1].set_title(title)
        fig.colorbar(im, ax=ax[1], fraction=0.046, pad=0.04)
        ax[2].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        ax[2].axis('off')
        ax[2].set_title("Overlay")
        plt.tight_layout()
        fig.savefig(output_path, dpi=150)
        plt.close(fig)


# ==================== Grad-CAM ====================

class GradCAMRunner(BaseModelRunner):  # ✅ FIXED
    def __init__(self, model_path, classes_path, target_layer='layer4', device=None):
        super().__init__(model_path, classes_path, device)
        self.activations = None
        self.gradients = None
        target_mod = dict(self.model.named_modules())[target_layer]
        target_mod.register_forward_hook(self._save_activation)
        target_mod.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inp, out):
        self.activations = out.detach().cpu().numpy()[0]

    def _save_gradient(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach().cpu().numpy()[0]

    def _make_grad_cam(self, inp, cls_idx):
        self.model.zero_grad()
        output = self.model(inp)
        score = output[0, cls_idx]
        score.backward()

        weights = self.gradients.mean(axis=(1, 2))
        cam = np.tensordot(weights, self.activations, axes=([0], [0]))
        cam = np.maximum(cam, 0)
        cam -= cam.min()
        cam /= (cam.max() + 1e-8)
        return cam

    def run(self, image_path, output_dir, alpha=0.5):
        img, inp = self._process_image(image_path)
        with torch.no_grad():
            logits = self.model(inp)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        cls_idx = probs.argmax()
        cam_map = self._make_grad_cam(inp, cls_idx)
        output_file_name = f"{Path(image_path).stem}_gradcam.png"
        output_file_path = output_dir / output_file_name
        self._generate_output_image(img, cam_map, output_file_path, 'Grad-CAM', alpha)
        return output_file_name, self.classes[cls_idx], float(probs[cls_idx])


# ==================== Integrated Gradients ====================

class IntegratedGradientsRunner(BaseModelRunner):
    def __init__(self, model_path, classes_path, device=None):
        super().__init__(model_path, classes_path, device)

    def _integrated_gradients(self, inp, cls_idx, baseline, steps):
        diff = inp - baseline
        total_grads = torch.zeros_like(inp)

        for i in range(1, steps + 1):
            alpha = float(i) / steps
            x = baseline + alpha * diff
            x.requires_grad_(True)
            x.retain_grad()
            self.model.zero_grad()
            out = self.model(x)
            score = out[0, cls_idx]
            score.backward()
            total_grads += x.grad

        avg_grads = total_grads / steps
        ig = diff * avg_grads
        ig_map = ig[0].detach().cpu().numpy().mean(axis=0)
        ig_map = np.maximum(ig_map, 0)
        ig_map -= ig_map.min()
        ig_map /= (ig_map.max() + 1e-8)
        return ig_map

    def run(self, image_path, output_dir, steps=50, alpha=0.5):
        img, inp = self._process_image(image_path)
        inp.requires_grad_(True)

        logits = self.model(inp)
        probs = torch.softmax(logits, dim=1).detach().cpu().numpy()[0]
        cls_idx = probs.argmax()

        baseline = torch.zeros_like(inp)
        ig_map = self._integrated_gradients(inp, cls_idx, baseline, steps)

        output_file_name = f"{Path(image_path).stem}_ig.png"
        output_file_path = output_dir / output_file_name
        self._generate_output_image(img, ig_map, output_file_path, 'Integrated Gradients', alpha)
        return output_file_name, self.classes[cls_idx], float(probs[cls_idx])


# ==================== SHAP ====================

class SHAPRunner(BaseModelRunner):
    def __init__(self, model_path, classes_path, device=None):
        super().__init__(model_path, classes_path, device)

        self.model_for_shap = self.model.to('cpu').eval()
        for m in self.model_for_shap.modules():
            if isinstance(m, nn.ReLU):
                m.inplace = False

        baseline = torch.zeros((1, 3, 224, 224), device='cpu')
        self.explainer = shap.GradientExplainer(self.model_for_shap, baseline)

    def run(self, image_path, output_dir, alpha=0.5):
        img, inp = self._process_image(image_path)
        inp_cpu = inp.to('cpu')

        with torch.no_grad():
            logits = self.model(inp_cpu)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        cls_idx = int(probs.argmax())
        cls_name = self.classes[cls_idx]

        shap_vals = self.explainer.shap_values(inp_cpu)
        cls_shap = shap_vals[cls_idx][0] if isinstance(shap_vals, list) else shap_vals[0]

        shap_map = np.mean(cls_shap, axis=0)
        shap_map = np.maximum(shap_map, 0)
        shap_map -= shap_map.min()
        shap_map /= (shap_map.max() + 1e-8)

        output_file_name = f"{Path(image_path).stem}_shap.png"
        output_file_path = output_dir / output_file_name
        self._generate_output_image(img, shap_map, output_file_path, 'SHAP', alpha)
        return output_file_name, cls_name, float(probs[cls_idx])


# ==================== API Routes ====================

@app.route('/')
def index():
    return jsonify({"message": "XAI API is running."})


@app.route('/cam', methods=['POST'])
def run_cam():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400
    uploaded_file = request.files['image']
    if uploaded_file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    temp_dir, input_dir, _ = get_session_temp_dirs()
    image_path = save_uploaded_file(uploaded_file, input_dir)

    try:
        runner = GradCAMRunner(MODEL_PATH, CLASSES_PATH)
        output_file_name, predicted_class, probability = runner.run(image_path, STATIC_IMG_DIR)
        return jsonify({
            "image_url": f"/static/{output_file_name}",
            "predicted_class": predicted_class,
            "probability": probability
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        shutil.rmtree(temp_dir)


@app.route('/ig', methods=['POST'])
def run_ig():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400
    uploaded_file = request.files['image']
    if uploaded_file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    temp_dir, input_dir, _ = get_session_temp_dirs()
    image_path = save_uploaded_file(uploaded_file, input_dir)

    try:
        runner = IntegratedGradientsRunner(MODEL_PATH, CLASSES_PATH)
        output_file_name, predicted_class, probability = runner.run(image_path, STATIC_IMG_DIR)
        return jsonify({
            "image_url": f"/static/{output_file_name}",
            "predicted_class": predicted_class,
            "probability": probability
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        shutil.rmtree(temp_dir)


@app.route('/shap', methods=['POST'])
def run_shap():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400
    uploaded_file = request.files['image']
    if uploaded_file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    temp_dir, input_dir, _ = get_session_temp_dirs()
    image_path = save_uploaded_file(uploaded_file, input_dir)

    try:
        runner = SHAPRunner(MODEL_PATH, CLASSES_PATH)
        output_file_name, predicted_class, probability = runner.run(image_path, STATIC_IMG_DIR)
        return jsonify({
            "image_url": f"/static/{output_file_name}",
            "predicted_class": predicted_class,
            "probability": probability
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        shutil.rmtree(temp_dir)


# ==================== Run App ====================

if __name__ == '__main__':
    if not Path(MODEL_PATH).exists():
        print(f"Warning: {MODEL_PATH} not found. Creating dummy model.")
        Path('model').mkdir(exist_ok=True)
        dummy_model = models.resnet50(weights=None)
        dummy_model.fc = nn.Linear(dummy_model.fc.in_features, 10)
        torch.save(dummy_model.state_dict(), MODEL_PATH)

    if not Path(CLASSES_PATH).exists():
        print(f"Warning: {CLASSES_PATH} not found. Creating dummy classes.")
        with open(CLASSES_PATH, 'w') as f:
            for i in range(10):
                f.write(f"class_{i}\n")

    app.run(host='0.0.0.0', port=5000)
