import os
import json
import torch
import random
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score, roc_curve
from sklearn.decomposition import PCA

DEFAULT_MODEL_NAME = "/root/outputs/SmolLM3-3B-SimPO-merged"
MODEL_NAME = os.environ.get("SIMPO_MODEL_NAME", DEFAULT_MODEL_NAME)
MAX_PAIRS = 500
MAX_LENGTH = 512
TEST_SIZE = 0.2
RANDOM_STATE = 42
DATASET_NAME = "argilla/ultrafeedback-binarized-preferences-cleaned"

random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)

output_dir = os.path.join("results", MODEL_NAME.replace("/", "_"))
os.makedirs(output_dir, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.truncation_side = "left"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=dtype,
    device_map="auto" if device == "cuda" else None,
)
model.eval()

if not (hasattr(model, "model") and hasattr(model.model, "layers")):
    raise ValueError("Unsupported model architecture.")

layers = model.model.layers
num_layers = len(layers)
hidden_size = model.config.hidden_size

dataset = load_dataset(DATASET_NAME, split="train")
dataset = dataset.select(range(min(MAX_PAIRS, len(dataset))))

activations = {}

def make_hook(layer_idx):
    def hook(module, inp, output):
        x = output[0] if isinstance(output, tuple) else output
        activations[layer_idx] = x.detach().float().cpu()
    return hook

hooks = [layers[i].register_forward_hook(make_hook(i)) for i in range(num_layers)]

def normalize_content(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for c in content:
            if isinstance(c, dict):
                if c.get("type") == "text":
                    parts.append(c.get("text", ""))
                elif "text" in c:
                    parts.append(str(c["text"]))
            else:
                parts.append(str(c))
        return "".join(parts)
    return str(content)

def sanitize_messages(messages):
    cleaned = []
    for msg in messages:
        if isinstance(msg, dict) and "role" in msg and "content" in msg:
            cleaned.append({"role": msg["role"], "content": normalize_content(msg["content"])})
    return cleaned

def get_response_text(messages):
    messages = sanitize_messages(messages)
    if len(messages) == 0 or messages[-1]["role"] != "assistant":
        raise ValueError("Expected final assistant message.")
    return messages[-1]["content"]

def build_context_and_response(messages):
    messages = sanitize_messages(messages)
    if len(messages) == 0 or messages[-1]["role"] != "assistant":
        raise ValueError("Expected final assistant message.")
    prefix_messages = messages[:-1]
    response_text = messages[-1]["content"]

    # Llama base models often do not define tokenizer.chat_template.
    # In that case, build a simple role-tagged transcript deterministically.
    is_llama_model = "llama" in MODEL_NAME.lower()
    has_chat_template = getattr(tokenizer, "chat_template", None) is not None

    if len(prefix_messages) == 0:
        prefix_text = ""
    elif is_llama_model and not has_chat_template:
        rendered_turns = []
        for msg in prefix_messages:
            role = msg["role"].strip().lower()
            if role == "system":
                role_tag = "System"
            elif role == "user":
                role_tag = "User"
            elif role == "assistant":
                role_tag = "Assistant"
            else:
                role_tag = role.title()
            rendered_turns.append(f"{role_tag}: {msg['content']}")
        prefix_text = "\n\n".join(rendered_turns) + "\n\nAssistant: "
    else:
        prefix_ids = tokenizer.apply_chat_template(
            prefix_messages,
            tokenize=True,
            add_generation_prompt=True,
        )
        prefix_text = tokenizer.decode(prefix_ids, skip_special_tokens=False)

    full_text = prefix_text + response_text
    return full_text, response_text

@torch.no_grad()
def extract_layerwise_mean_pooled_vectors(messages):
    full_text, response_text = build_context_and_response(messages)

    full_enc = tokenizer(
        full_text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH,
        padding=False,
        add_special_tokens=False,
    )
    resp_enc = tokenizer(
        response_text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH,
        padding=False,
        add_special_tokens=False,
    )

    full_ids = full_enc["input_ids"][0]
    resp_ids = resp_enc["input_ids"][0]

    if resp_ids.numel() == 0:
        raise ValueError("Empty response tokens.")

    resp_len = resp_ids.shape[0]
    seq_len = full_ids.shape[0]

    if resp_len > seq_len:
        raise ValueError("Response longer than full sequence.")

    response_start = seq_len - resp_len
    response_ids_in_full = full_ids[response_start:]

    if not torch.equal(response_ids_in_full.cpu(), resp_ids.cpu()):
        tail_len = min(resp_len, seq_len)
        response_start = seq_len - tail_len

    input_ids = full_enc["input_ids"].to(model.get_input_embeddings().weight.device)
    attention_mask = full_enc["attention_mask"].to(model.get_input_embeddings().weight.device)

    activations.clear()
    _ = model(input_ids=input_ids, attention_mask=attention_mask)

    vectors = []
    for layer_idx in range(num_layers):
        h = activations[layer_idx][0]
        pooled = h[response_start:].mean(dim=0).numpy().astype(np.float32)
        vectors.append(pooled)

    return vectors

pair_records = []
skip_reasons = {}

for item in tqdm(dataset, desc="Collecting activations", unit="pair"):
    try:
        chosen_vecs = extract_layerwise_mean_pooled_vectors(item["chosen"])
        rejected_vecs = extract_layerwise_mean_pooled_vectors(item["rejected"])
        pair_records.append({"chosen": chosen_vecs, "rejected": rejected_vecs})
    except Exception as e:
        reason = str(e)
        skip_reasons[reason] = skip_reasons.get(reason, 0) + 1

for h in hooks:
    h.remove()

if len(pair_records) < 10:
    raise RuntimeError(f"Too few valid pairs collected ({len(pair_records)}). Top skip reasons: {skip_reasons}")

pair_indices = np.arange(len(pair_records))
rng = np.random.RandomState(RANDOM_STATE)
rng.shuffle(pair_indices)

test_count = max(1, int(round(TEST_SIZE * len(pair_indices))))
test_pair_idx = sorted(pair_indices[:test_count].tolist())
train_pair_idx = sorted(pair_indices[test_count:].tolist())

layer_metrics = []
layer_predictions = {}
layer_probabilities = {}

for layer_idx in tqdm(range(num_layers), desc="Training probes", unit="layer"):
    X_train, y_train = [], []
    X_test, y_test = [], []

    for i in train_pair_idx:
        X_train.append(pair_records[i]["chosen"][layer_idx])
        y_train.append(1)
        X_train.append(pair_records[i]["rejected"][layer_idx])
        y_train.append(0)

    for i in test_pair_idx:
        X_test.append(pair_records[i]["chosen"][layer_idx])
        y_test.append(1)
        X_test.append(pair_records[i]["rejected"][layer_idx])
        y_test.append(0)

    X_train = np.stack(X_train)
    y_train = np.array(y_train)
    X_test = np.stack(X_test)
    y_test = np.array(y_test)

    train_mask = np.isfinite(X_train).all(axis=1)
    test_mask = np.isfinite(X_test).all(axis=1)

    X_train = X_train[train_mask]
    y_train = y_train[train_mask]
    X_test = X_test[test_mask]
    y_test = y_test[test_mask]

    probe = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=5000,
            solver="lbfgs",
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
    )
    probe.fit(X_train, y_train)

    y_pred = probe.predict(X_test)
    y_prob = probe.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auroc = roc_auc_score(y_test, y_prob)
    auprc = average_precision_score(y_test, y_prob)
    coef_norm = float(np.linalg.norm(probe.named_steps["logisticregression"].coef_))

    layer_metrics.append(
        {
            "layer": layer_idx,
            "accuracy": float(acc),
            "f1": float(f1),
            "auroc": float(auroc),
            "auprc": float(auprc),
            "coef_norm": coef_norm,
            "num_train_samples": int(len(y_train)),
            "num_test_samples": int(len(y_test)),
        }
    )

    layer_predictions[layer_idx] = {
        "y_test": y_test.tolist(),
        "y_pred": y_pred.tolist(),
    }
    layer_probabilities[layer_idx] = y_prob.tolist()

with open(os.path.join(output_dir, "layer_metrics.json"), "w") as f:
    json.dump(layer_metrics, f, indent=2)

with open(os.path.join(output_dir, "layer_predictions.json"), "w") as f:
    json.dump(layer_predictions, f, indent=2)

with open(os.path.join(output_dir, "layer_probabilities.json"), "w") as f:
    json.dump(layer_probabilities, f, indent=2)

best_layer = max(layer_metrics, key=lambda x: x["auroc"])["layer"]

X_train_best, y_train_best = [], []
X_test_best, y_test_best = [], []

for i in train_pair_idx:
    X_train_best.append(pair_records[i]["chosen"][best_layer])
    y_train_best.append(1)
    X_train_best.append(pair_records[i]["rejected"][best_layer])
    y_train_best.append(0)

for i in test_pair_idx:
    X_test_best.append(pair_records[i]["chosen"][best_layer])
    y_test_best.append(1)
    X_test_best.append(pair_records[i]["rejected"][best_layer])
    y_test_best.append(0)

X_train_best = np.stack(X_train_best)
y_train_best = np.array(y_train_best)
X_test_best = np.stack(X_test_best)
y_test_best = np.array(y_test_best)

train_mask = np.isfinite(X_train_best).all(axis=1)
test_mask = np.isfinite(X_test_best).all(axis=1)

X_train_best = X_train_best[train_mask]
y_train_best = y_train_best[train_mask]
X_test_best = X_test_best[test_mask]
y_test_best = y_test_best[test_mask]

best_probe = make_pipeline(
    StandardScaler(),
    LogisticRegression(
        max_iter=5000,
        solver="lbfgs",
        class_weight="balanced",
        random_state=RANDOM_STATE,
    ),
)
best_probe.fit(X_train_best, y_train_best)
best_probs = best_probe.predict_proba(X_test_best)[:, 1]
fpr, tpr, _ = roc_curve(y_test_best, best_probs)

layers_x = [m["layer"] for m in layer_metrics]
accuracy_y = [m["accuracy"] for m in layer_metrics]
f1_y = [m["f1"] for m in layer_metrics]
auroc_y = [m["auroc"] for m in layer_metrics]
auprc_y = [m["auprc"] for m in layer_metrics]
coefnorm_y = [m["coef_norm"] for m in layer_metrics]

plt.figure(figsize=(8, 5))
plt.plot(layers_x, accuracy_y, marker="o", label="Accuracy")
plt.plot(layers_x, f1_y, marker="o", label="F1")
plt.plot(layers_x, auroc_y, marker="o", label="AUROC")
plt.plot(layers_x, auprc_y, marker="o", label="AUPRC")
plt.xlabel("Layer")
plt.ylabel("Score")
plt.title(f"Layerwise probe performance: {MODEL_NAME}")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "layerwise_probe_metrics.png"), dpi=200)
plt.close()

plt.figure(figsize=(8, 5))
plt.plot(layers_x, coefnorm_y, marker="o")
plt.xlabel("Layer")
plt.ylabel("L2 norm")
plt.title(f"Probe coefficient norm by layer: {MODEL_NAME}")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "layerwise_coef_norm.png"), dpi=200)
plt.close()

chosen_probs = best_probs[y_test_best == 1]
rejected_probs = best_probs[y_test_best == 0]

plt.figure(figsize=(8, 5))
plt.hist(rejected_probs, bins=25, alpha=0.7, label="Rejected", density=True)
plt.hist(chosen_probs, bins=25, alpha=0.7, label="Chosen", density=True)
plt.xlabel("Predicted probability of chosen")
plt.ylabel("Density")
plt.title(f"Best layer probability separation (layer {best_layer})")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "best_layer_probability_hist.png"), dpi=200)
plt.close()

plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, label=f"Layer {best_layer} ROC")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False positive rate")
plt.ylabel("True positive rate")
plt.title(f"ROC curve at best layer {best_layer}")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "best_layer_roc_curve.png"), dpi=200)
plt.close()

scaler = best_probe.named_steps["standardscaler"]
X_test_scaled = scaler.transform(X_test_best)
pca = PCA(n_components=2, random_state=RANDOM_STATE)
X_test_pca = pca.fit_transform(X_test_scaled)

pca_payload = {
    "best_layer": int(best_layer),
    "y_test": y_test_best.tolist(),
    "pc1": X_test_pca[:, 0].tolist(),
    "pc2": X_test_pca[:, 1].tolist(),
    "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
    "components": pca.components_.tolist(),
}
with open(os.path.join(output_dir, "best_layer_pca.json"), "w") as f:
    json.dump(pca_payload, f, indent=2)

plt.figure(figsize=(7, 6))
plt.scatter(X_test_pca[y_test_best == 0, 0], X_test_pca[y_test_best == 0, 1], alpha=0.7, s=18, label="Rejected")
plt.scatter(X_test_pca[y_test_best == 1, 0], X_test_pca[y_test_best == 1, 1], alpha=0.7, s=18, label="Chosen")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title(f"PCA of held-out activations at best layer {best_layer}")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "best_layer_pca.png"), dpi=200)
plt.close()

summary = {
    "model_name": MODEL_NAME,
    "dataset": DATASET_NAME,
    "max_pairs_requested": MAX_PAIRS,
    "max_pairs_used": len(pair_records),
    "max_length": MAX_LENGTH,
    "num_layers": num_layers,
    "hidden_size": hidden_size,
    "train_pairs": len(train_pair_idx),
    "test_pairs": len(test_pair_idx),
    "best_layer_by_auroc": int(best_layer),
    "best_metrics": next(m for m in layer_metrics if m["layer"] == best_layer),
    "skip_reasons": skip_reasons,
    "saved_files": [
        "layer_metrics.json",
        "layer_predictions.json",
        "layer_probabilities.json",
        "best_layer_pca.json",
        "layerwise_probe_metrics.png",
        "layerwise_coef_norm.png",
        "best_layer_probability_hist.png",
        "best_layer_roc_curve.png",
        "best_layer_pca.png",
    ],
}

with open(os.path.join(output_dir, "summary.json"), "w") as f:
    json.dump(summary, f, indent=2)

print(json.dumps(summary, indent=2))
print(f"All outputs saved to: {output_dir}/")
