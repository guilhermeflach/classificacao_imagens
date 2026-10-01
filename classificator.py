from PIL import Image
import torch
from transformers import CLIPModel, CLIPProcessor

device = "cuda" if torch.cuda.is_available() else "cpu"

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
model.eval()

CATEGORIAS = {
    "exterior": [
        "an exterior photo of a house or building facade",
        "a street view of a neighborhood or building",
        "an aerial drone photo of a property or land",
        "outdoor patio, garden, backyard or swimming pool",
    ],
    "interior": [
        "an indoor photo of a living room, bedroom, kitchen, or bathroom",
        "an interior room of an apartment or house",
    ],
    "outros": [
        "an architectural floor plan, 2D blueprint, or layout drawing",
        "a text document, logo, contact card, or map",
    ],
}

flat_labels = []
label_map = []
for cat, prompts in CATEGORIAS.items():
    for p in prompts:
        flat_labels.append(p)
        label_map.append(cat)

with torch.no_grad():
    text_inputs = processor(
        text=flat_labels, padding=True, return_tensors="pt"
    ).to(device)
    text_features = model.get_text_features(**text_inputs)
    if not isinstance(text_features, torch.Tensor):
        text_features = getattr(
            text_features, "pooler_output", text_features[0]
        )
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)


def classificar_foto(caminho_foto: str, threshold_exterior: float = 0.5):
    try:
        image = Image.open(caminho_foto).convert("RGB")
    except Exception as e:
        print(f"Erro ao abrir {caminho_foto}: {e}")
        return False, {}

    with torch.no_grad():
        image_inputs = processor(images=image, return_tensors="pt").to(device)
        image_features = model.get_image_features(**image_inputs)
        if not isinstance(image_features, torch.Tensor):
            image_features = getattr(
                image_features, "pooler_output", image_features[0]
            )
        image_features = image_features / image_features.norm(
            dim=-1, keepdim=True
        )

        similarity = (
            (image_features @ text_features.T) * model.logit_scale.exp()
        )
        probs = similarity.softmax(dim=-1).squeeze(0)

    scores = {"exterior": 0.0, "interior": 0.0, "outros": 0.0}
    for prob, cat in zip(probs, label_map):
        scores[cat] += prob.item()

    is_exterior = (
        scores["exterior"] > threshold_exterior
        and scores["exterior"] > scores["interior"]
    )
    return is_exterior, scores


# Teste
caminho = input('Nome da imagem:')+'.jpg'

eh_ext, scores = classificar_foto(caminho)

print(f"É exterior? {eh_ext}")
if scores:
    print(
        f"Exterior: {scores['exterior']:.1%} | Interior: {scores['interior']:.1%} | Outros: {scores['outros']:.1%}"
    )