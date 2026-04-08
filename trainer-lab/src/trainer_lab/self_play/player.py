"""Self-play player with MCTS + neural network for training data generation."""

from __future__ import annotations

import copy
import logging
import math
import random
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from trainer_lab.config import SelfPlayConfig
from trainer_lab.data.encoder import board_to_tensor
from trainer_lab.models.resnet import PolicyValueResNet

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# GameState — lightweight board representation for self-play
# ---------------------------------------------------------------------------


class GameState:
    """Minimal board state for MCTS self-play."""

    __slots__ = ("board_size", "win_length", "board", "current_player", "last_move", "move_count")

    def __init__(self, board_size: int = 15, win_length: int | None = None) -> None:
        self.board_size = board_size
        self.win_length = win_length if win_length is not None else (4 if board_size <= 5 else 5)
        self.board: list[list[int]] = [[0] * board_size for _ in range(board_size)]
        self.current_player = 1  # 1 or 2
        self.last_move: tuple[int, int] | None = None
        self.move_count = 0

    def legal_moves(self) -> list[int]:
        """Return flat indices of all empty cells."""
        bs = self.board_size
        return [r * bs + c for r in range(bs) for c in range(bs) if self.board[r][c] == 0]

    def apply_move(self, flat_idx: int) -> GameState:
        """Return a new GameState with the move applied."""
        bs = self.board_size
        r, c = divmod(flat_idx, bs)
        new = GameState.__new__(GameState)
        new.board_size = bs
        new.win_length = self.win_length
        new.board = [row[:] for row in self.board]
        new.board[r][c] = self.current_player
        new.current_player = 3 - self.current_player
        new.last_move = (r, c)
        new.move_count = self.move_count + 1
        return new

    def is_terminal(self) -> tuple[bool, int]:
        """Check if game is over. Returns (is_over, winner).

        winner=0 means draw; winner=1 or 2 means that player won.
        """
        if self.last_move is None:
            return False, 0
        r, c = self.last_move
        player = self.board[r][c]
        bs = self.board_size
        wl = self.win_length

        for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
            count = 1
            for s in range(1, wl):
                nr, nc = r + dr * s, c + dc * s
                if 0 <= nr < bs and 0 <= nc < bs and self.board[nr][nc] == player:
                    count += 1
                else:
                    break
            for s in range(1, wl):
                nr, nc = r - dr * s, c - dc * s
                if 0 <= nr < bs and 0 <= nc < bs and self.board[nr][nc] == player:
                    count += 1
                else:
                    break
            if count >= wl:
                return True, player

        if self.move_count >= bs * bs:
            return True, 0  # draw
        return False, 0

    def to_encoder_dict(self) -> dict:
        """Convert to the dict format expected by board_to_tensor()."""
        return {
            "board_size": self.board_size,
            "board": self.board,
            "current_player": self.current_player,
            "last_move": list(self.last_move) if self.last_move else None,
        }


# ---------------------------------------------------------------------------
# MCTSNode
# ---------------------------------------------------------------------------


class MCTSNode:
    """A single node in the MCTS search tree."""

    __slots__ = ("parent", "move", "prior", "visit_count", "value_sum", "children")

    def __init__(self, parent: MCTSNode | None, move: int, prior: float) -> None:
        self.parent = parent
        self.move = move
        self.prior = prior
        self.visit_count = 0
        self.value_sum = 0.0
        self.children: list[MCTSNode] = []

    @property
    def q_value(self) -> float:
        return self.value_sum / self.visit_count if self.visit_count > 0 else 0.0

    def ucb_score(self, c_puct: float = 1.5) -> float:
        parent_visits = self.parent.visit_count if self.parent else 1
        u = c_puct * self.prior * math.sqrt(parent_visits) / (1 + self.visit_count)
        return self.q_value + u

    def best_child(self, c_puct: float = 1.5) -> MCTSNode:
        return max(self.children, key=lambda c: c.ucb_score(c_puct))

    def is_leaf(self) -> bool:
        return len(self.children) == 0


# ---------------------------------------------------------------------------
# MCTS search
# ---------------------------------------------------------------------------


def _expand_node(
    node: MCTSNode,
    state: GameState,
    model: PolicyValueResNet,
    device: torch.device,
) -> float:
    """Expand a leaf node using the neural network. Returns the value estimate."""
    tensor = board_to_tensor(state.to_encoder_dict()).unsqueeze(0).to(device)
    with torch.no_grad():
        policy_logits, value_tensor = model(tensor)

    value = value_tensor.item()
    logits = policy_logits.squeeze(0).cpu()
    legal = state.legal_moves()

    if not legal:
        return value

    # Mask illegal moves and apply softmax
    bs = state.board_size
    mask = torch.full((logits.shape[0],), float("-inf"))
    for m in legal:
        r, c = divmod(m, bs)
        mask[r * 16 + c] = 0.0  # policy uses 16x16 grid
    probs = F.softmax(logits + mask, dim=0)

    for m in legal:
        r, c = divmod(m, bs)
        p = probs[r * 16 + c].item()
        node.children.append(MCTSNode(parent=node, move=m, prior=p))

    return value


def mcts_search(
    root_state: GameState,
    model: PolicyValueResNet,
    device: torch.device,
    num_simulations: int = 400,
    c_puct: float = 1.5,
    dirichlet_alpha: float = 0.03,
    dirichlet_weight: float = 0.25,
) -> tuple[list[float], float]:
    """Run MCTS from root_state. Returns (policy_vector, root_value).

    policy_vector has length board_size^2 with visit count proportions.
    """
    bs = root_state.board_size
    root = MCTSNode(parent=None, move=-1, prior=1.0)
    root_value = _expand_node(root, root_state, model, device)

    # Add Dirichlet noise to root priors for exploration
    if root.children:
        noise = np.random.dirichlet([dirichlet_alpha] * len(root.children))
        for child, n in zip(root.children, noise):
            child.prior = (1 - dirichlet_weight) * child.prior + dirichlet_weight * n

    # Build a state mapping for traversal
    for _ in range(num_simulations):
        node = root
        state = root_state

        # Selection: walk down tree
        while not node.is_leaf():
            node = node.best_child(c_puct)
            state = state.apply_move(node.move)

        # Check terminal
        terminal, winner = state.is_terminal()
        if terminal:
            if winner == 0:
                value = 0.0
            elif winner == state.current_player:
                value = 1.0
            else:
                value = -1.0
        else:
            # Expansion
            value = _expand_node(node, state, model, device)

        # Backup: propagate value up the tree (flip sign at each level)
        while node is not None:
            node.visit_count += 1
            node.value_sum += value
            value = -value
            node = node.parent

    # Build policy vector from root visit counts
    policy = [0.0] * (bs * bs)
    total_visits = sum(c.visit_count for c in root.children)
    if total_visits > 0:
        for child in root.children:
            policy[child.move] = child.visit_count / total_visits

    return policy, root_value


# ---------------------------------------------------------------------------
# SelfPlayPlayer
# ---------------------------------------------------------------------------


class SelfPlayPlayer:
    """Uses MCTS + neural network to generate self-play training games."""

    def __init__(
        self,
        model: PolicyValueResNet,
        config: SelfPlayConfig | None = None,
        device: torch.device | None = None,
    ) -> None:
        self.model = model
        self.config = config or SelfPlayConfig()
        self.device = device or torch.device("cpu")
        self.model.to(self.device)
        self.model.eval()

    def play_game(self, board_size: int = 15, win_length: int | None = None) -> list[dict]:
        """Play a single self-play game using MCTS.

        Returns list of position records with 256-padded policy targets:
        ``{board_size, board, current_player, last_move, policy, value}``
        """
        wl = win_length if win_length is not None else (4 if board_size <= 5 else 5)
        state = GameState(board_size, wl)
        positions: list[dict] = []
        move_count = 0
        # Temperature exploration: proportional for first N moves, then greedy
        tau_threshold = max(4, board_size)  # 5 for ttt5, 15 for gomoku15

        while True:
            terminal, winner = state.is_terminal()
            if terminal:
                break

            policy_flat, _ = mcts_search(
                state,
                self.model,
                self.device,
                num_simulations=self.config.simulations,
            )

            # Convert board_size^2 policy to 256-padded (16x16) format
            policy_256 = [0.0] * 256
            for idx, prob in enumerate(policy_flat):
                if prob > 0:
                    r, c = divmod(idx, board_size)
                    policy_256[r * 16 + c] = prob

            # Store position before making the move
            positions.append({
                "board_size": board_size,
                "board": [row[:] for row in state.board],
                "current_player": state.current_player,
                "last_move": list(state.last_move) if state.last_move else None,
                "policy": policy_256,
                "value": 0.0,  # filled after game ends
                "source": "self_play",
            })

            # Temperature-based move selection
            if move_count < tau_threshold:
                # Sample proportionally (tau=1)
                total = sum(policy_flat)
                if total > 0:
                    move = random.choices(range(len(policy_flat)), weights=policy_flat, k=1)[0]
                else:
                    move = random.choice(state.legal_moves())
            else:
                # Greedy (tau->0)
                move = max(range(len(policy_flat)), key=lambda i: policy_flat[i])

            state = state.apply_move(move)
            move_count += 1

        # Fill game outcome values
        _, winner = state.is_terminal()
        for pos in positions:
            if winner == 0:
                pos["value"] = 0.0
            elif pos["current_player"] == winner:
                pos["value"] = 1.0
            else:
                pos["value"] = -1.0

        return positions

    def generate_games(
        self,
        num_games: int | None = None,
        board_size: int = 15,
        win_length: int | None = None,
    ) -> list[dict]:
        """Generate multiple self-play games and collect all positions."""
        n = num_games if num_games is not None else self.config.games
        all_positions: list[dict] = []
        for g in range(n):
            positions = self.play_game(board_size=board_size, win_length=win_length)
            all_positions.extend(positions)
            if (g + 1) % max(1, n // 10) == 0:
                logger.info("Self-play: %d/%d games (%d positions)", g + 1, n, len(all_positions))
        return all_positions
