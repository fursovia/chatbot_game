from typing import Union, Optional, Tuple

import torch

DISCRIMINATOR_MODELS_PARAMS = {
    "clickbait": {
        "path": "models/clickbait_classifier_head.pt",
        "class_size": 2,
        "embed_size": 1024,
        "class_vocab": {"non_clickbait": 0, "clickbait": 1},
        "default_class": 1,
        "pretrained_model": "gpt2-medium",
    },
    "sentiment": {
        "path": "models/SST_classifier_head.pt",
        "class_size": 5,
        "embed_size": 1024,
        "class_vocab": {"very_positive": 2, "very_negative": 3},
        "default_class": 3,
        "pretrained_model": "gpt2-medium",
    },
}


class ClassificationHead(torch.nn.Module):
    """Classification Head for transformer encoders"""

    def __init__(self,
                 class_size: int,
                 embed_size: int):
        """
        :param class_size: number of labels
        :param embed_size: embeddings vector size
        """
        super(ClassificationHead, self).__init__()
        self.class_size = class_size
        self.embed_size = embed_size
        self.mlp = torch.nn.Linear(embed_size, class_size)

    def forward(self,
                hidden_state: torch.Tensor):
        logits = self.mlp(hidden_state)
        return logits


def get_classifier(
        name: Optional[str],
        class_label: Union[str, int],
        device: str,
        classifiers_dir: str) -> Tuple[Optional[ClassificationHead], Optional[int]]:
    if name is None:
        return None, None

    params = DISCRIMINATOR_MODELS_PARAMS[name]
    classifier = ClassificationHead(class_size=params['class_size'],
                                    embed_size=params['embed_size']).to(device)

    resolved_archive_file = classifiers_dir + params["path"]
    classifier.load_state_dict(torch.load(resolved_archive_file, map_location=device))
    classifier.eval()

    if isinstance(class_label, str):
        if class_label in params["class_vocab"]:
            label_id = params["class_vocab"][class_label]
        else:
            label_id = params["default_class"]
    elif isinstance(class_label, int):
        if class_label in set(params["class_vocab"].values()):
            label_id = class_label
        else:
            label_id = params["default_class"]
    else:
        label_id = params["default_class"]

    return classifier, label_id
