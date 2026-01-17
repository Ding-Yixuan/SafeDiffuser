import torch
import torch.nn as nn
from typing import Sequence, Optional, List, Iterable


def _get_activation(act_name: Optional[str]):
    if not act_name:
        return None
    n = act_name.lower()
    if n in ("relu", "r"):
        return nn.ReLU()
    if n in ("tanh",):
        return nn.Tanh()
    if n in ("elu",):
        return nn.ELU()
    if n in ("sigmoid",):
        return nn.Sigmoid()
    return nn.ReLU()


def _build_mlp(sizes: Sequence[int], activation: Optional[nn.Module]):
    layers: List[nn.Module] = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        # only add activation between hidden layers
        if activation is not None and i < len(sizes) - 2:
            layers.append(activation)
    return nn.Sequential(*layers)


class CBFNetwork(nn.Module):
    """
    - 保留共享特征提取 trunk 和 certificate head
    - 返回一个 certificate 张量，形状为 (batch, output_dim)
    - 构造函数兼容常见参数名：output_dim / num_outputs, hidden_layers / hidden_sizes

    如果只需要单值 barrier，传入
    output_dim=1（默认）。测试中如果传入 output_dim>1，将返回相应形状张量以
    保证兼容性
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: Optional[int] = None,
        hidden_layers: Optional[Iterable[int]] = None,
        activation: str = "tanh",
        num_outputs: Optional[int] = None,
        hidden_sizes: Optional[Iterable[int]] = None,
    ):
        super().__init__()
        # 参数兼容处理
        if hidden_layers is None and hidden_sizes is not None:
            hidden_layers = tuple(hidden_sizes)
        if hidden_layers is None:
            # 使用更适合 CBF 的默认隐藏层
            hidden_layers = (256, 256)
        self.hidden_layers = tuple(hidden_layers)

        out_size = output_dim if output_dim is not None else num_outputs
        if out_size is None:
            out_size = 1

        act = _get_activation(activation)

        # 构建 trunk
        sizes = [input_dim] + list(self.hidden_layers) if len(self.hidden_layers) > 0 else [input_dim]
        self.trunk = _build_mlp(sizes, act)
        final_dim = sizes[-1]

        # certificate head: 支持多输出以兼容测试
        self.certificate_head = nn.Linear(final_dim, int(out_size))
        nn.init.normal_(self.certificate_head.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.certificate_head.bias)

        self._output_dim = int(out_size)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """返回 certificate 张量，形状 (batch, output_dim)。
        如果需要 1-d 向量，可在外部调用 .squeeze(-1)
        """
        x = obs.float()
        last = self.trunk(x)
        cert = self.certificate_head(last)
        return cert