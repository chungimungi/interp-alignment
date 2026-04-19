"""Map model ids to UltraFeedback linear-probe datasets."""

DATASET_DPO_ORPO = "argilla/ultrafeedback-binarized-preferences-cleaned"
DATASET_GRPO = "argilla/ultrafeedback-multi-binarized-preferences-cleaned"
DATASET_KTO = "argilla/ultrafeedback-binarized-preferences-cleaned-kto"


def infer_dataset_for_model(model_id: str) -> str:
    """Infer dataset from model id (DPO/ORPO, GRPO, KTO suffixes in org checkpoint names)."""
    m = model_id.lower()
    if "grpo" in m:
        return DATASET_GRPO
    if "kto" in m:
        return DATASET_KTO
    if "dpo" in m or "orpo" in m:
        return DATASET_DPO_ORPO
    return DATASET_DPO_ORPO


def is_kto_dataset(dataset_id: str) -> bool:
    return dataset_id.rstrip("/").endswith("-kto") or "preferences-cleaned-kto" in dataset_id
