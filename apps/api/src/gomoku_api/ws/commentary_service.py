"""Human-friendly move commentary for the legacy game UI.

This is the first step toward a richer coach / hint / narrator mode:
we use structured tactical analysis plus model afterstate values, and
then retrieve a phrase template that explains the move in plain language.
"""

from __future__ import annotations

import random
from typing import Any

from gomoku_api.ws.predict_service import (
    _count_double_threat_responses,
    _evaluate_afterstate_values,
    _find_immediate_move,
    _get_model,
    _list_immediate_wins,
)


def _resolve_variant_spec(board: list[int], variant: str | None) -> tuple[str, int, int]:
    resolved_variant = (variant or "").strip()
    if not resolved_variant:
        if len(board) == 9:
            resolved_variant = "ttt3"
        elif len(board) == 25:
            resolved_variant = "ttt5"
        else:
            resolved_variant = f"gomoku{int(len(board) ** 0.5)}"

    if resolved_variant == "ttt3":
        return resolved_variant, 3, 3
    if resolved_variant == "ttt5":
        return resolved_variant, 5, 4
    if resolved_variant.startswith("gomoku"):
        size = int(resolved_variant.replace("gomoku", ""))
        return resolved_variant, size, 5
    raise ValueError(f"Unsupported commentary variant: {resolved_variant}")


def _move_label(move: int, board_size: int) -> str:
    if move < 0:
        return "?"
    row, col = divmod(move, board_size)
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    return f"{letters[col]}{row + 1}"


def _subject_labels(actor: str) -> tuple[str, str]:
    actor_key = str(actor or "player").strip().lower()
    if actor_key in {"bot", "model", "ai"}:
        return "Бот", "вы"
    return "Вы", "соперник"


def _seeded_choice(options: list[str], seed_payload: tuple[Any, ...]) -> str:
    rng = random.Random(repr(seed_payload))
    return rng.choice(options)


_PHRASES: dict[str, dict[str, list[str]]] = {
    "coach": {
        "found_win": [
            "{subject} великолепно завершили атаку: {move} сразу выигрывает партию.",
            "{subject} увидели точное добивание. Ход {move} закрывает игру.",
        ],
        "missed_win": [
            "{subject} упустили немедленную победу. Завершал партию ход {best_move}.",
            "{subject} уже держали матовый удар, но не дожали. Сильнейшим был {best_move}.",
        ],
        "blocked_threat": [
            "{subject} вовремя заметили угрозу и закрыли её ходом {move}.",
            "{subject} точно защитились: {move} снимает прямую опасность.",
        ],
        "missed_block": [
            "{subject} пропустили обязательную защиту. Нужно было отвечать {best_move}.",
            "{subject} не закрыли критическую угрозу. Сейчас правильный щит был на {best_move}.",
        ],
        "blunder_allow_win": [
            "{subject} открыли сопернику победу в один ход. Это очень опасно.",
            "{subject} допустили тактический зевок: после этого ответа у соперника есть немедленное добивание.",
        ],
        "create_fork": [
            "{subject} захватили инициативу: после {move} у позиции появляется двойная угроза.",
            "{subject} построили форсирующее давление. Ход {move} даёт сразу несколько победных идей.",
        ],
        "allow_fork": [
            "{subject} ослабили позицию и позволили сопернику развить двойную угрозу.",
            "{subject} оставили слишком много контригры. Соперник получает сильные форсирующие ответы.",
        ],
        "seize_advantage": [
            "{subject} захватили преимущество. Ход {move} усиливает давление и улучшает позицию.",
            "{subject} перехватили инициативу. После {move} позиция смотрится заметно увереннее.",
        ],
        "lost_advantage": [
            "{subject} утратили лидерство. Сильнее выглядел ход {best_move}.",
            "{subject} недожали позицию. После {move} преимущество стало скромнее, чем могло быть.",
        ],
        "under_pressure": [
            "{subject} пока остаётесь под давлением. Здесь нужна очень точная защита.",
            "Позиция после {move} всё ещё неприятная: {opponent} сохраняет инициативу.",
        ],
        "solid_move": [
            "{subject} сделали спокойный рабочий ход {move}. Позиция остаётся управляемой.",
            "{subject} сыграли аккуратно. Это не взрывной ход, но он держит структуру позиции.",
        ],
    },
    "emotional": {
        "found_win": [
            "Блестяще! {subject} нашли удар {move} и закрыли партию.",
            "Вот это точность: {move} и позиция рассыпается в пользу {subject_lc}.",
        ],
        "missed_win": [
            "Ох, победа уже была в руках. Нужно было бить {best_move}.",
            "Здесь можно было заканчивать красиво, но решающий удар {best_move} остался незамеченным.",
        ],
        "blocked_threat": [
            "Отличная выдержка: {subject} увидели опасность и всё перекрыли.",
            "Хорошая защита. Паники нет — угроза закрыта.",
        ],
        "missed_block": [
            "Опасный момент: защита на {best_move} была обязательной.",
            "Сейчас запахло бедой — угроза осталась без ответа.",
        ],
        "blunder_allow_win": [
            "Это уже красная тревога: соперник получает победу в один ход.",
            "Очень болезненный зевок — после этого соперник может сразу наказать.",
        ],
        "create_fork": [
            "{subject} разогнали атаку! После {move} позиция начинает трещать.",
            "Мощно. {move} раскручивает сразу несколько угроз.",
        ],
        "allow_fork": [
            "Темп уходит сопернику: позиция стала заметно тревожнее.",
            "Неприятно: соперник получил слишком много воздуха для контратаки.",
        ],
        "seize_advantage": [
            "{subject} захватили преимущество и диктуют темп.",
            "Инициатива пошла в руки {subject_lc}: ход выглядит очень уверенно.",
        ],
        "lost_advantage": [
            "{subject} отпустили позицию. Момент для давления был сильнее на {best_move}.",
            "Лидерство качнулось назад — это был шанс сыграть жёстче.",
        ],
        "under_pressure": [
            "Позиция скрипит: {subject_lc} всё ещё приходится обороняться.",
            "Свободы мало — соперник продолжает давить.",
        ],
        "solid_move": [
            "Неплохо. Без фейерверка, но позиция держится.",
            "Спокойный ход, без лишнего риска.",
        ],
    },
    "hint": {
        "found_win": [
            "Подсказка: это точный победный ход. {move} завершает партию.",
        ],
        "missed_win": [
            "Подсказка: здесь была немедленная победа. Обратите внимание на {best_move}.",
        ],
        "blocked_threat": [
            "Подсказка: защита выбрана правильно. {move} закрывает прямую угрозу.",
        ],
        "missed_block": [
            "Подсказка: сначала нужно было защищаться. Ключевая клетка — {best_move}.",
        ],
        "blunder_allow_win": [
            "Подсказка: этот ход оставляет сопернику победу в один ответ. Ищите более безопасную защиту.",
        ],
        "create_fork": [
            "Подсказка: ход создаёт двойное давление и несколько победных идей сразу.",
        ],
        "allow_fork": [
            "Подсказка: после этого ответа соперник получает слишком много сильных продолжений.",
        ],
        "seize_advantage": [
            "Подсказка: хороший ход. Он заметно улучшает оценку позиции.",
        ],
        "lost_advantage": [
            "Подсказка: позиция ещё играбельна, но ход {best_move} давал больше давления.",
        ],
        "under_pressure": [
            "Подсказка: позиция остаётся тяжёлой. Ищите самый безопасный оборонительный ресурс.",
        ],
        "solid_move": [
            "Подсказка: ход рабочий, но не форсирующий. Ищите способы усилить инициативу.",
        ],
    },
}


def _advantage_label(score: float) -> str:
    if score >= 0.55:
        return "сильное преимущество"
    if score >= 0.20:
        return "инициатива"
    if score <= -0.55:
        return "тяжёлая оборона"
    if score <= -0.20:
        return "давление соперника"
    return "примерное равновесие"


def analyze_move_commentary(
    board_before: list[int],
    move: int,
    current: int,
    *,
    variant: str | None = None,
    style: str = "coach",
    actor: str = "player",
) -> dict[str, Any]:
    resolved_variant, board_size, win_len = _resolve_variant_spec(board_before, variant)
    style_key = str(style or "coach").strip().lower()
    if style_key not in _PHRASES:
        style_key = "coach"

    legal = [idx for idx, cell in enumerate(board_before) if cell == 0]
    subject, opponent_label = _subject_labels(actor)
    subject_lc = subject.lower()
    move_name = _move_label(move, board_size)

    if move < 0 or move >= len(board_before) or board_before[move] != 0:
        return {
            "actor": actor,
            "style": style_key,
            "category": "invalid",
            "mood": "danger",
            "text": f"{subject} выбрали недопустимый ход. Попробуйте другую клетку.",
            "move": move,
            "moveLabel": move_name,
            "ply": sum(1 for cell in board_before if cell != 0) + 1,
        }

    after = list(board_before)
    after[move] = current
    opponent = 2 if current == 1 else 1

    model = _get_model(resolved_variant)
    value_scores: dict[int, float] = {}
    if model is not None and legal:
        value_scores = _evaluate_afterstate_values(model, board_before, current, board_size, legal)

    best_move = max(
        legal,
        key=lambda candidate: (float(value_scores.get(candidate, 0.0)), -abs(candidate - move)),
    ) if legal else move
    chosen_score = float(value_scores.get(move, 0.0))
    best_score = float(value_scores.get(best_move, chosen_score))
    score_gap = best_score - chosen_score

    immediate_win = _find_immediate_move(board_before, board_size, win_len, current)
    opponent_immediate_before = _list_immediate_wins(board_before, board_size, win_len, opponent)
    opponent_immediate_after = _list_immediate_wins(after, board_size, win_len, opponent)
    self_immediate_after = _list_immediate_wins(after, board_size, win_len, current)
    opponent_forks_after = 0 if opponent_immediate_after else _count_double_threat_responses(after, board_size, win_len, opponent)

    if immediate_win is not None and move == immediate_win:
        category, mood = "found_win", "positive"
    elif immediate_win is not None and move != immediate_win:
        category, mood = "missed_win", "danger"
    elif opponent_immediate_before and move in opponent_immediate_before and not opponent_immediate_after:
        category, mood = "blocked_threat", "positive"
    elif opponent_immediate_before and move not in opponent_immediate_before:
        category, mood = "missed_block", "danger"
    elif opponent_immediate_after:
        category, mood = "blunder_allow_win", "danger"
    elif len(self_immediate_after) >= 2:
        category, mood = "create_fork", "positive"
    elif opponent_forks_after > 0:
        category, mood = "allow_fork", "warning"
    elif model is not None and chosen_score >= 0.30 and score_gap <= 0.10:
        category, mood = "seize_advantage", "positive"
    elif model is not None and score_gap >= 0.35:
        category, mood = "lost_advantage", "warning"
    elif model is not None and chosen_score <= -0.25:
        category, mood = "under_pressure", "warning"
    else:
        category, mood = "solid_move", "neutral"

    suggested_move = (
        immediate_win
        if category == "missed_win"
        else (opponent_immediate_before[0] if category == "missed_block" and opponent_immediate_before else best_move)
    )
    if suggested_move is None:
        suggested_move = best_move

    display_best_move = suggested_move if category in {"missed_win", "missed_block"} else best_move
    best_move_label = _move_label(display_best_move, board_size)
    templates = _PHRASES[style_key].get(category) or _PHRASES["coach"]["solid_move"]
    text = _seeded_choice(
        templates,
        (tuple(board_before), move, current, resolved_variant, style_key, actor, category),
    ).format(
        subject=subject,
        subject_lc=subject_lc,
        opponent=opponent_label,
        move=move_name,
        best_move=best_move_label,
    )

    risk_level = "high" if mood == "danger" else "medium" if mood == "warning" else "low"
    tags = [category]
    if opponent_immediate_before:
        tags.append("forced_defense")
    if opponent_immediate_after:
        tags.append("allow_immediate_loss")
    if len(self_immediate_after) >= 2:
        tags.append("double_threat")
    if model is not None:
        tags.append("model_eval")

    return {
        "actor": actor,
        "style": style_key,
        "category": category,
        "mood": mood,
        "riskLevel": risk_level,
        "text": text,
        "move": move,
        "moveLabel": move_name,
        "bestMove": display_best_move,
        "bestMoveLabel": best_move_label,
        "suggestedMove": suggested_move,
        "chosenScore": round(chosen_score, 4),
        "bestScore": round(best_score, 4),
        "scoreGap": round(score_gap, 4),
        "advantageLabel": _advantage_label(chosen_score),
        "opponentThreatsBefore": len(opponent_immediate_before),
        "opponentThreatsAfter": len(opponent_immediate_after),
        "forcingThreatsAfter": len(self_immediate_after),
        "opponentForkResponses": int(opponent_forks_after),
        "tags": tags,
        "ply": sum(1 for cell in board_before if cell != 0) + 1,
        "variant": resolved_variant,
    }
