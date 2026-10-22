from __future__ import annotations

import torch
from torch import nn


def build_groups(mean_shape: torch.Tensor, num_groups: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Partition vertices into ``num_groups`` disjoint, spatially compact groups.

    Recursive median bisection along the axis of largest spread of the mean
    face shape. Disjointness matters: an overlapping vertex would reappear
    clean in a visible tube and leak a masked tube's content.

    Returns ``groups (G, n_max)`` (padded with vertex 0) and ``valid (G, n_max)``.
    """
    vertices = mean_shape.shape[0]
    leaves: list[torch.Tensor] = []

    def split(indices: torch.Tensor, parts: int) -> None:
        if parts == 1:
            leaves.append(indices)
            return
        left_parts = parts // 2
        spread = mean_shape[indices].var(dim=0)
        order = mean_shape[indices, int(spread.argmax())].argsort()
        indices = indices[order]
        cut = round(len(indices) * left_parts / parts)
        split(indices[:cut], left_parts)
        split(indices[cut:], parts - left_parts)

    split(torch.arange(vertices), num_groups)
    n_max = max(len(leaf) for leaf in leaves)
    groups = torch.zeros(num_groups, n_max, dtype=torch.long)
    valid = torch.zeros(num_groups, n_max, dtype=torch.bool)
    for g, leaf in enumerate(leaves):
        groups[g, : len(leaf)] = leaf
        valid[g, : len(leaf)] = True
    return groups, valid


def motion_groups(mesh: torch.Tensor, num_groups: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Partition vertices into ``num_groups`` disjoint groups of vertices that
    *move together*, instead of ``build_groups``' spatial proximity.

    Affinity = positive part of the Pearson correlation between two vertices'
    inter-frame displacements (x and y pooled) over the given training clips
    ``mesh (N, T, V, 2)``. The vertex set is split by recursive spectral
    bisection: the Fiedler vector of the affinity graph's normalised Laplacian
    orders the vertices and the cut is by count, exactly as in
    ``build_groups``, so groups are balanced (6-7 vertices for 16 groups).
    """
    velocity = (mesh[:, 1:] - mesh[:, :-1]).permute(2, 0, 1, 3).reshape(mesh.shape[2], -1).double()
    velocity = velocity - velocity.mean(dim=1, keepdim=True)
    velocity = velocity / velocity.norm(dim=1, keepdim=True).clamp_min(1e-12)
    affinity = (velocity @ velocity.T).clamp_min(0)
    leaves: list[torch.Tensor] = []

    def split(indices: torch.Tensor, parts: int) -> None:
        if parts == 1:
            leaves.append(indices)
            return
        left_parts = parts // 2
        a = affinity[indices][:, indices]
        d = a.sum(dim=1).clamp_min(1e-12).rsqrt()
        laplacian = torch.eye(len(indices), dtype=a.dtype) - d[:, None] * a * d[None, :]
        _, vectors = torch.linalg.eigh(laplacian)
        indices = indices[(vectors[:, 1] * d).argsort()]
        cut = round(len(indices) * left_parts / parts)
        split(indices[:cut], left_parts)
        split(indices[cut:], parts - left_parts)

    split(torch.arange(mesh.shape[2]), num_groups)
    n_max = max(len(leaf) for leaf in leaves)
    groups = torch.zeros(num_groups, n_max, dtype=torch.long)
    valid = torch.zeros(num_groups, n_max, dtype=torch.bool)
    for g, leaf in enumerate(leaves):
        groups[g, : len(leaf)] = leaf
        valid[g, : len(leaf)] = True
    return groups, valid


def anatomical_groups(
    mean_shape: torch.Tensor, region_of: dict[int, str], jaw_parts: int = 3
) -> tuple[torch.Tensor, torch.Tensor]:
    """One token group per hand-annotated facial part (``dimp/landmark_regions.json``)
    instead of ``build_groups``' blind geometric bisection, which can mix an eye
    and its brow in one tube or split the mouth across two.

    Each pupil joins its eye (gaze is eye-relative motion); nose_bridge (4
    points) joins nose_tip; the 33-point jaw contour is cut into ``jaw_parts``
    pieces along x of the mean shape so one outline region does not get a
    single oversized token. Groups stay disjoint (no masked-content leak).
    Returns ``groups (G, n_max)`` (padded with vertex 0) and ``valid``.
    """
    merge = {"left_pupil": "left_eye", "right_pupil": "right_eye", "nose_bridge": "nose_tip"}
    members: dict[str, list[int]] = {}
    for vertex in range(mean_shape.shape[0]):
        name = region_of[vertex]
        members.setdefault(merge.get(name, name), []).append(vertex)
    leaves: list[torch.Tensor] = []
    for name in sorted(members):
        indices = torch.tensor(members[name])
        if name == "jaw_contour" and jaw_parts > 1:
            ordered = indices[mean_shape[indices, 0].argsort()]
            leaves += list(ordered.tensor_split(jaw_parts))
        else:
            leaves.append(indices)
    n_max = max(len(leaf) for leaf in leaves)
    groups = torch.zeros(len(leaves), n_max, dtype=torch.long)
    valid = torch.zeros(len(leaves), n_max, dtype=torch.bool)
    for g, leaf in enumerate(leaves):
        groups[g, : len(leaf)] = leaf
        valid[g, : len(leaf)] = True
    return groups, valid


class TubeLayout(nn.Module):
    """Turns ``(B, T, V, 2)`` landmark clips into spatio-temporal tube tokens.

    A tube is one vertex group over ``tube_frames`` consecutive frames. Token
    index is ``k = s * G + g`` (temporal chunk ``s``, group ``g``).

    Per token it provides
      * ``points``  ``(B, K, n, l, 2)``: coordinates / ``point_std``, padding zeroed;
      * ``centers`` ``(B, K, 2)``: tube mean, mean-centred over the clip's tubes
        (removes any global offset, as in the paper) and / ``center_std``;
      * ``motion``  ``(B, K, n, l-1, 2)``: point-wise inter-frame displacement
        ``p[t+1] - p[t]`` / ``motion_std``. Vertex identity is fixed by the
        landmark topology, so correspondences are exact and no learned
        correspondence model is involved.
    The three scale constants are dataset statistics stored as buffers.
    """

    def __init__(
        self, num_groups: int, group_size: int, tube_frames: int, frames: int, num_vertices: int = 106
    ) -> None:
        super().__init__()
        if frames % tube_frames != 0:
            raise ValueError("frames must be divisible by tube_frames")
        self.num_groups = num_groups
        self.group_size = group_size
        self.tube_frames = tube_frames
        self.frames = frames
        self.chunks = frames // tube_frames
        self.num_tokens = self.chunks * num_groups
        self.register_buffer("groups", torch.zeros(num_groups, group_size, dtype=torch.long))
        self.register_buffer("valid", torch.zeros(num_groups, group_size, dtype=torch.bool))
        self.register_buffer("scales", torch.ones(3))  # point_std, center_std, motion_std
        token = torch.arange(self.num_tokens)
        self.register_buffer("chunk_index", token // num_groups, persistent=False)
        self.register_buffer("group_index", token % num_groups, persistent=False)

    def set_groups(self, groups: torch.Tensor, valid: torch.Tensor) -> None:
        self.groups.copy_(groups)
        self.valid.copy_(valid)

    @torch.no_grad()
    def raw_tokens(self, mesh: torch.Tensor) -> dict[str, torch.Tensor]:
        """Tokens with unit scales (used to measure the dataset statistics)."""
        return self._tokens(mesh, torch.ones(3, device=mesh.device))

    def tokens(self, mesh: torch.Tensor) -> dict[str, torch.Tensor]:
        return self._tokens(mesh, self.scales)

    def _tokens(self, mesh: torch.Tensor, scales: torch.Tensor) -> dict[str, torch.Tensor]:
        batch, frames, _, _ = mesh.shape
        if frames != self.frames:
            raise ValueError(f"expected {self.frames} frames, got {frames}")
        valid = self.valid  # (G, n)
        points = mesh[:, :, self.groups] / scales[0]  # (B, T, G, n, 2)
        points = points * valid[None, None, :, :, None]
        points = points.reshape(batch, self.chunks, self.tube_frames, self.num_groups, self.group_size, 2)
        points = points.permute(0, 1, 3, 4, 2, 5)  # (B, S, G, n, l, 2)

        count = valid.sum(dim=1).view(1, 1, -1, 1) * self.tube_frames  # (1,1,G,1)
        centers = points.sum(dim=(3, 4)) / count  # (B, S, G, 2)
        centers = centers - centers.mean(dim=(1, 2), keepdim=True)
        centers = centers / scales[1]

        motion = points[..., 1:, :] - points[..., :-1, :]
        motion_valid = valid[None, None, :, :, None, None]
        motion = motion * motion_valid / scales[2]

        k = self.num_tokens
        return {
            "points": points.reshape(batch, k, self.group_size, self.tube_frames, 2),
            "centers": centers.reshape(batch, k, 2),
            "motion": motion.reshape(batch, k, self.group_size, self.tube_frames - 1, 2),
        }

    def point_valid(self) -> torch.Tensor:
        """``(G, n)`` validity mask expanded to ``(K, n)`` in token order."""
        return self.valid[self.group_index]
