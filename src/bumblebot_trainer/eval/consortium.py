import torch

from .utils import autocast, get_model_embeddings, stack_positions


class PolicyConsortium:
    """Expert consortium: several experts groups merging policies and value predictions
    based on weights from an expert routing model.

    Used to test different experts in their corresponding positions on full-game evaluations.
    """
    ROUTER_CLASSES = {
        'activity': ('strategical', 'tactical'),
        'evaluation': ('winning', 'drawing', 'losing'),
    }

    def __init__(self, router_base, router_heads, experts, expert_groups):
        self.router_base = router_base
        self.router_heads = router_heads
        self.experts = experts
        self.expert_groups = tuple(expert_groups)

    @torch.inference_mode()
    def predict_move_ids(self, batch, device):
        tokens, extra = stack_positions(batch, device)
        legal_ids = [
            torch.as_tensor(item['legal_ids'], dtype=torch.long, device=device)
            for item in batch
        ]
        policy_votes = [torch.zeros(ids.numel(), device=device) for ids in legal_ids]
        group_scale = 1.0 / len(self.expert_groups)

        route_weights = {}
        router_logits = None
        if 'activity' in self.expert_groups or 'evaluation' in self.expert_groups:
            with autocast(device):
                router_embeddings, _ = get_model_embeddings(self.router_base, tokens, extra)
                router_logits, _ = self.router_heads(router_embeddings)

        if 'activity' in self.expert_groups:
            route_weights['activity'] = torch.softmax(router_logits['activity'], dim=-1)
        if 'evaluation' in self.expert_groups:
            route_weights['evaluation'] = torch.softmax(router_logits['evaluation'], dim=-1)

        for group in self.expert_groups:
            for expert_name, expert in self.experts[group].items():
                if group == 'phase':
                    indices = [
                        i for i, item in enumerate(batch)
                        if item['phase'] == expert_name
                    ]
                    weights = None
                else:
                    indices = list(range(len(batch)))
                    class_idx = self.ROUTER_CLASSES[group].index(expert_name)
                    weights = route_weights[group][:, class_idx] * group_scale

                if not indices:
                    continue

                index_tensor = torch.tensor(indices, dtype=torch.long, device=device)
                expert_extra = {key: value[index_tensor] for key, value in extra.items()}
                with autocast(device):
                    expert_embeddings, _ = get_model_embeddings(
                        expert, tokens[index_tensor], expert_extra
                    )
                    logits = expert.policy_head(expert_embeddings, None).logits

                for local_idx, batch_idx in enumerate(indices):
                    move_ids = legal_ids[batch_idx]
                    move_probabilities = torch.softmax(logits[local_idx, move_ids], dim=-1)
                    weight = group_scale if weights is None else weights[batch_idx]
                    policy_votes[batch_idx] += move_probabilities * weight

        return [
            int(legal_ids[index][votes.argmax()].item())
            for index, votes in enumerate(policy_votes)
        ]

