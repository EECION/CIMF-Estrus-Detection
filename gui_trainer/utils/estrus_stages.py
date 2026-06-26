from typing import Dict, List


class EstrusStageRegistry:
    STAGES = ["pre_estrus", "estrus", "post_estrus"]
    STAGE_TO_IDX = {name: idx for idx, name in enumerate(STAGES)}
    IDX_TO_STAGE = {idx: name for idx, name in enumerate(STAGES)}

    @classmethod
    def get_class_names(cls) -> List[str]:
        return list(cls.STAGES)

    @classmethod
    def get_num_classes(cls) -> int:
        return len(cls.STAGES)

    @classmethod
    def get_label(cls, stage_name: str) -> int:
        return cls.STAGE_TO_IDX[stage_name]

    @classmethod
    def get_stage(cls, label: int) -> str:
        return cls.IDX_TO_STAGE[label]

    @classmethod
    def get_display_names(cls) -> Dict[str, str]:
        return {
            "pre_estrus": "Pre-Estrus",
            "estrus": "Estrus",
            "post_estrus": "Post-Estrus",
        }
