import os
import json
import torch
import random
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score, roc_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.decomposition import PCA


MODEL_NAME = "Qwen/Qwen3-4B" # CHANGE THIS for Base / SFT / DPO
DATASET_NAME = "argilla/ultrafeedback-binarized-preferences-cleaned"

MAX_PAIRS = 5000
MAX_LENGTH = 512
BATCH_SIZE = 8        
K_FOLDS = 5          
RANDOM_STATE = 42

def configure_plot_style():
    plt.rcParams.update({
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.family": "sans-serif",
        "font.size": 13,
        "axes.titlesize": 16,
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 11,
        "axes.linewidth": 1.2,
        "lines.linewidth": 2.0,
        "lines.markersize": 6,
        "savefig.bbox": "tight",
    })

def set_layer_ticks(ax, layers):
    desired_ticks = [10, 20, 30]
    ticks = [t for t in desired_ticks if min(layers) <= t <= max(layers)]
    if ticks:
        ax.set_xticks(ticks)
    ax.set_xlim(min(layers) - 0.5, max(layers) + 0.5)

random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)

output_dir = os.path.join("results", MODEL_NAME.replace("/", "_"))
os.makedirs(output_dir, exist_ok=True)
configure_plot_style()

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

print(f"Loading Tokenizer: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"  

print(f"Loading Model: {MODEL_NAME}")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=dtype,
    device_map="auto" if device == "cuda" else None,
)
model.eval()

layers = model.model.layers
num_layers = len(layers)
hidden_size = model.config.hidden_size

def normalize_content(content):
    if isinstance(content, str): return content
    if isinstance(content, list):
        return "".join([str(c.get("text", c)) if isinstance(c, dict) else str(c) for c in content])
    return str(content)

def sanitize_messages(messages):
    return [{"role": msg["role"], "content": normalize_content(msg["content"])} for msg in messages if "role" in msg]

def build_prompt_text(messages):
    messages = sanitize_messages(messages)
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    except:
        # fallback for base models
        rendered = [f"{msg['role'].title()}: {msg['content']}" for msg in messages]
        return "\n\n".join(rendered)

print(f"Loading Dataset: {DATASET_NAME}")
dataset = load_dataset(DATASET_NAME, split="train")
dataset = dataset.select(range(min(MAX_PAIRS, len(dataset))))

chosen_texts, rejected_texts = [], []
for item in dataset:
    try:
        chosen_texts.append(build_prompt_text(item["chosen"]))
        rejected_texts.append(build_prompt_text(item["rejected"]))
    except Exception as e:
        continue

valid_pairs = len(chosen_texts)
print(f"Prepared {valid_pairs} valid chosen/rejected pairs.")

activations = {}
def make_hook(layer_idx):
    def hook(module, inp, output):
        x = output[0] if isinstance(output, tuple) else output
        # because we use left-padding, the final true token is always exactly at index -1
        last_token_rep = x[:, -1, :].detach().float().cpu().numpy()
        activations[layer_idx] = last_token_rep
    return hook

hooks = [layers[i].register_forward_hook(make_hook(i)) for i in range(num_layers)]

def extract_features(texts, desc="Extracting"):
    features_by_layer = {i: [] for i in range(num_layers)}
    
    for i in tqdm(range(0, len(texts), BATCH_SIZE), desc=desc):
        batch_texts = texts[i:i+BATCH_SIZE]
        inputs = tokenizer(
            batch_texts, 
            return_tensors="pt", 
            truncation=True, 
            max_length=MAX_LENGTH, 
            padding=True
        ).to(device)
        
        with torch.no_grad():
            _ = model(**inputs)
            
        for layer_idx in range(num_layers):
            features_by_layer[layer_idx].append(activations[layer_idx])
            
    for layer_idx in range(num_layers):
        features_by_layer[layer_idx] = np.concatenate(features_by_layer[layer_idx], axis=0)
    return features_by_layer

chosen_features = extract_features(chosen_texts, desc="Extracting Chosen")
rejected_features = extract_features(rejected_texts, desc="Extracting Rejected")

for h in hooks:
    h.remove()

layer_metrics = []
layer_predictions = {}
layer_probabilities = {}

cv = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=RANDOM_STATE)

for layer_idx in tqdm(range(num_layers), desc="Training Contrastive Probes"):
    X_c = chosen_features[layer_idx]
    X_r = rejected_features[layer_idx]
    
    # contrastive setup: delta = Chosen - Rejected.
    X_diff = X_c - X_r
    
    # create symmetrical labels to avoid probe bias
    # 1 indicates the vector is (Chosen - Rejected)
    # 0 indicates the vector is (Rejected - Chosen)
    X_sym = np.vstack([X_diff, -X_diff])
    y_sym = np.concatenate([np.ones(valid_pairs), np.zeros(valid_pairs)])
    
    valid_mask = np.isfinite(X_sym).all(axis=1)
    X_sym = X_sym[valid_mask]
    y_sym = y_sym[valid_mask]

    oof_preds = np.zeros(len(y_sym))
    oof_probs = np.zeros(len(y_sym))
    coef_norms = []
    
    for train_idx, test_idx in cv.split(X_sym, y_sym):
        X_train, y_train = X_sym[train_idx], y_sym[train_idx]
        X_test, y_test_fold = X_sym[test_idx], y_sym[test_idx]
        
        probe = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=5000, solver="lbfgs", class_weight="balanced", random_state=RANDOM_STATE)
        )
        probe.fit(X_train, y_train)
        
        oof_preds[test_idx] = probe.predict(X_test)
        oof_probs[test_idx] = probe.predict_proba(X_test)[:, 1]
        coef_norms.append(float(np.linalg.norm(probe.named_steps["logisticregression"].coef_)))
        
    acc = accuracy_score(y_sym, oof_preds)
    f1 = f1_score(y_sym, oof_preds)
    auroc = roc_auc_score(y_sym, oof_probs)
    auprc = average_precision_score(y_sym, oof_probs)
    
    layer_metrics.append({
        "layer": layer_idx,
        "accuracy": float(acc),
        "f1": float(f1),
        "auroc": float(auroc),
        "auprc": float(auprc),
        "coef_norm": float(np.mean(coef_norms)),
    })
    
    layer_predictions[layer_idx] = {"y_test": y_sym.tolist(), "y_pred": oof_preds.tolist()}
    layer_probabilities[layer_idx] = oof_probs.tolist()

with open(os.path.join(output_dir, "layer_metrics.json"), "w") as f:
    json.dump(layer_metrics, f, indent=2)
with open(os.path.join(output_dir, "layer_predictions.json"), "w") as f:
    json.dump(layer_predictions, f, indent=2)
with open(os.path.join(output_dir, "layer_probabilities.json"), "w") as f:
    json.dump(layer_probabilities, f, indent=2)

best_layer = max(layer_metrics, key=lambda x: x["auroc"])["layer"]
X_c_best = chosen_features[best_layer]
X_r_best = rejected_features[best_layer]

X_pca_input = np.vstack([X_r_best, X_c_best])
y_pca_labels = np.concatenate([np.zeros(len(X_r_best)), np.ones(len(X_c_best))])

valid_mask_pca = np.isfinite(X_pca_input).all(axis=1)
X_pca_input = X_pca_input[valid_mask_pca]
y_pca_labels = y_pca_labels[valid_mask_pca]

pca = PCA(n_components=2, random_state=RANDOM_STATE)
X_scaled = StandardScaler().fit_transform(X_pca_input)
X_pca = pca.fit_transform(X_scaled)

pca_payload = {
    "best_layer": int(best_layer),
    "y_test": y_pca_labels.tolist(),
    "pc1": X_pca[:, 0].tolist(),
    "pc2": X_pca[:, 1].tolist(),
    "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
}
with open(os.path.join(output_dir, "best_layer_pca.json"), "w") as f:
    json.dump(pca_payload, f, indent=2)

print("Generating PDF plots...")

layers_list = [m["layer"] for m in layer_metrics]
acc_list = [m["accuracy"] for m in layer_metrics]
f1_list = [m["f1"] for m in layer_metrics]
auroc_list = [m["auroc"] for m in layer_metrics]
auprc_list = [m["auprc"] for m in layer_metrics]
coef_norm_list = [m["coef_norm"] for m in layer_metrics]

# 1) layerwise metrics
fig, ax = plt.subplots(figsize=(15, 10))
ax.plot(layers_list, acc_list, marker="o", label="Accuracy")
ax.plot(layers_list, f1_list, marker="^", label="F1")
ax.plot(layers_list, auroc_list, marker="P", label="AUROC")
ax.plot(layers_list, auprc_list, marker="s", label="AUPRC")
ax.set_xlabel("Layer")
ax.set_ylabel("Score")
set_layer_ticks(ax, layers_list)
ax.grid(alpha=0.3)
ax.legend(frameon=False)
fig.savefig(os.path.join(output_dir, "layerwise_probe_metrics.pdf"))
plt.close(fig)

# 2) coefficient norm by layer
fig, ax = plt.subplots(figsize=(15, 10))
ax.plot(layers_list, coef_norm_list, marker="o")
ax.set_xlabel("Layer")
ax.set_ylabel("L2 Norm")
set_layer_ticks(ax, layers_list)
ax.grid(alpha=0.3)
fig.savefig(os.path.join(output_dir, "layerwise_coef_norm.pdf"))
plt.close(fig)

# 3) best-layer probability histograms
y_prob_best = np.array(layer_probabilities[best_layer])
y_test_best = np.array(layer_predictions[best_layer]["y_test"])

chosen_probs = y_prob_best[y_test_best == 1]
rejected_probs = y_prob_best[y_test_best == 0]

fig, ax = plt.subplots(figsize=(14, 10))
ax.hist(rejected_probs, bins=24, alpha=0.7, density=True, label="Reverse Direction (-Δx)")
ax.hist(chosen_probs, bins=24, alpha=0.7, density=True, label="Preference Direction (+Δx)")
ax.set_xlabel("Predicted Probability of Preferred Direction")
ax.set_ylabel("Density")
ax.grid(alpha=0.25)
ax.legend(frameon=False)
fig.savefig(os.path.join(output_dir, "best_layer_probability_hist.pdf"))
plt.close(fig)

# 4) roc curve
fpr, tpr, _ = roc_curve(y_test_best, y_prob_best)
fig, ax = plt.subplots(figsize=(10, 10))
ax.plot(fpr, tpr, label=f"Layer {best_layer}")
ax.plot([0, 1], [0, 1], linestyle="--", linewidth=3, label="Random baseline")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.grid(alpha=0.3)
ax.legend(frameon=False)
fig.savefig(os.path.join(output_dir, "best_layer_roc_curve.pdf"))
plt.close(fig)

# 5) best-layer pca scatter
pc1 = np.array(pca_payload["pc1"])
pc2 = np.array(pca_payload["pc2"])
pca_y = np.array(pca_payload["y_test"])

fig, ax = plt.subplots(figsize=(12, 10))
ax.scatter(pc1[pca_y == 0], pc2[pca_y == 0], alpha=0.7, s=80, label="Rejected")
ax.scatter(pc1[pca_y == 1], pc2[pca_y == 1], alpha=0.7, s=80, label="Chosen")
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.grid(alpha=0.25)
ax.legend(frameon=False)
fig.savefig(os.path.join(output_dir, "best_layer_pca.pdf"))
plt.close(fig)

print(f"Done! Results and strictly styled PDFs saved to: {output_dir}/")
