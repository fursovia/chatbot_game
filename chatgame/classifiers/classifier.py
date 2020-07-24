import torch


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
