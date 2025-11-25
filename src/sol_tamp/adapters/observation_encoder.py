"""Flatten GraphInstance observations for neural networks."""

import numpy as np
from gymnasium.spaces import GraphInstance
from relational_structs import GroundAtom


class ObservationEncoder:
    """Converts GraphInstance observations to flat vectors."""

    def __init__(self, max_nodes: int = 100, feature_dim: int = 64):
        self.max_nodes = max_nodes
        self.feature_dim = feature_dim
        self.atom_to_idx: dict[str, int] = {}
        self._next_atom_idx = 0

    def encode(
        self,
        obs: GraphInstance | np.ndarray,
        current_atoms: set[GroundAtom] | None = None,
        goal_atoms: set[GroundAtom] | None = None,
    ) -> np.ndarray:
        if isinstance(obs, GraphInstance):
            flat_obs = self._flatten_graph(obs)
        elif isinstance(obs, np.ndarray):
            flat_obs = obs.flatten().astype(np.float32)
        else:
            flat_obs = np.array(obs, dtype=np.float32).flatten()

        if current_atoms is not None or goal_atoms is not None:
            symbolic_features = self._encode_atoms(current_atoms, goal_atoms)
            flat_obs = np.concatenate([flat_obs, symbolic_features])

        return flat_obs

    def _flatten_graph(self, graph: GraphInstance) -> np.ndarray:
        nodes = graph.nodes if hasattr(graph, "nodes") else graph["nodes"]
        num_nodes = nodes.shape[0]

        if num_nodes <= self.max_nodes:
            padded = np.zeros((self.max_nodes, self.feature_dim), dtype=np.float32)
            padded[:num_nodes, : nodes.shape[1]] = nodes[:, : self.feature_dim]
        else:
            padded = nodes[: self.max_nodes, : self.feature_dim]

        return padded.flatten()

    def _encode_atoms(
        self,
        current_atoms: set[GroundAtom] | None,
        goal_atoms: set[GroundAtom] | None,
    ) -> np.ndarray:
        max_atoms = 50
        vector = np.zeros(max_atoms * 2, dtype=np.float32)

        if current_atoms:
            for atom in current_atoms:
                idx = self._get_atom_index(str(atom))
                if idx < max_atoms:
                    vector[idx] = 1.0

        if goal_atoms:
            for atom in goal_atoms:
                idx = self._get_atom_index(str(atom))
                if idx < max_atoms:
                    vector[max_atoms + idx] = 1.0

        return vector

    def _get_atom_index(self, atom_str: str) -> int:
        if atom_str not in self.atom_to_idx:
            self.atom_to_idx[atom_str] = self._next_atom_idx
            self._next_atom_idx += 1
        return self.atom_to_idx[atom_str]

    def get_output_dim(self, include_atoms: bool = False) -> int:
        base_dim = self.max_nodes * self.feature_dim
        return base_dim + (100 if include_atoms else 0)
