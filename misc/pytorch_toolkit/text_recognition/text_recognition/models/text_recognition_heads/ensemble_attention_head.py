from text_recognition.models.text_recognition_heads.attention_based_2d import TextRecognitionHeadAttention
import torch


class EnsembleAttentionHead(torch.nn.Module):
    def __init__(self, out_size, configuration, **kwargs):
        super().__init__()
        self.num_heads = len(configuration)
        for i in range(self.num_heads):
            setattr(self, f'head_{i}', TextRecognitionHeadAttention(out_size, **configuration[i]))

    def forward(self, features, texts=None):
        decoder_outs, semantic_info = [], []
        for i in range(self.num_heads):
            out, _, *sem_inf = getattr(self, f'head_{i}')(features[i], texts)
            if sem_inf:
                semantic_info.append(sem_inf[0])
            decoder_outs.append(out)
        decoder_outs = torch.mean(torch.stack(decoder_outs), dim=0).to(features[0].device)
        classes = torch.max(decoder_outs, dim=2)[1]
        if semantic_info:
            semantic_info = torch.mean(torch.stack(semantic_info), dim=0).to(features[0].device)
            return decoder_outs, classes, semantic_info
        return decoder_outs, classes
