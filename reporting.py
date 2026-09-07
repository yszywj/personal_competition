"""Durable text/JSON/CSV reporting and dependency-free SVG charts."""

from __future__ import annotations

import csv
import html
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


def _finite(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _atomic_text(path: Path, content: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def _json_default(value: Any):
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


class TrainingReporter:
    """Append one durable row and refresh charts after every completed round."""

    ROUND_FIELDS = (
        "round",
        "score",
        "K",
        "T",
        "completed",
        "steps",
        "elapsed_seconds",
        "red_launched",
        "red_alive",
        "red_lost",
        "red_intercepted",
        "interceptors_launched",
        "interceptors_successful",
        "interceptors_multi_kill",
        "max_kills_by_one_interceptor",
        "agent_return_sum",
        "agent_return_mean",
        "ppo_updates",
        "ppo_transitions",
        "policy_loss",
        "value_loss",
        "entropy",
        "approx_kl",
        "clip_fraction",
        "explained_variance",
        "learning_rate",
        "entropy_coef",
        "epochs_ran",
        "early_stopped",
        "action_left",
        "action_stop",
        "action_right",
        "action_switches",
    )
    PPO_FIELDS = (
        "round",
        "update_count",
        "transition_count",
        "samples",
        "trajectories",
        "rollout_environment_steps",
        "policy_loss",
        "value_loss",
        "entropy",
        "approx_kl",
        "clip_fraction",
        "explained_variance",
        "learning_rate",
        "entropy_coef",
        "epochs_ran",
        "early_stopped",
    )

    def __init__(self, result_dir: Path) -> None:
        self.result_dir = Path(result_dir)
        self.result_dir.mkdir(parents=True, exist_ok=True)
        self.rows: list[dict[str, Any]] = []
        self.score_text_path = self.result_dir / "round_scores.txt"
        self.score_csv_path = self.result_dir / "round_scores.csv"
        self.metrics_jsonl_path = self.result_dir / "round_metrics.jsonl"
        self.ppo_csv_path = self.result_dir / "ppo_metrics.csv"
        self.score_text_path.write_text(
            "R9 PPO final official score for every completed round\n",
            encoding="utf-8",
        )
        with self.score_csv_path.open("w", encoding="utf-8", newline="") as stream:
            csv.DictWriter(stream, fieldnames=self.ROUND_FIELDS).writeheader()
        with self.ppo_csv_path.open("w", encoding="utf-8", newline="") as stream:
            csv.DictWriter(stream, fieldnames=self.PPO_FIELDS).writeheader()
        self.metrics_jsonl_path.write_text("", encoding="utf-8")
        # Even an interruption before round 1 leaves the promised chart files.
        self._write_charts()

    def write_run_config(self, config: Mapping[str, Any]) -> None:
        _atomic_text(
            self.result_dir / "run_config.json",
            json.dumps(config, ensure_ascii=False, indent=2, default=_json_default) + "\n",
        )

    def record_round(
        self,
        *,
        round_index: int,
        summary: Mapping[str, Any],
        reward_metrics: Mapping[str, Any],
        policy,
        elapsed_seconds: float,
        action_counts: Mapping[int, int],
        action_switches: int,
    ) -> dict[str, Any]:
        score = summary.get("score") or {}
        red = summary.get("red") or {}
        interception = summary.get("interception") or {}
        ppo = dict(getattr(policy, "last_metrics", {}) or {})
        row = {
            "round": int(round_index),
            "score": _finite(score.get("score")),
            "K": _finite(score.get("K")),
            "T": _finite(score.get("T")),
            "completed": bool(score.get("completed", False)),
            "steps": int(summary.get("steps_executed", 0)),
            "elapsed_seconds": _finite(elapsed_seconds),
            "red_launched": int(red.get("launched") or 0),
            "red_alive": int(red.get("alive") or 0),
            "red_lost": int(red.get("lost") or 0),
            "red_intercepted": int(
                interception.get("red_missiles_intercepted") or 0
            ),
            "interceptors_launched": int(
                interception.get("interceptors_launched") or 0
            ),
            "interceptors_successful": int(
                interception.get("successful_interceptors") or 0
            ),
            "interceptors_multi_kill": int(
                interception.get("multi_kill_interceptors") or 0
            ),
            "max_kills_by_one_interceptor": int(
                interception.get("max_kills_by_one_interceptor") or 0
            ),
            "agent_return_sum": _finite(reward_metrics.get("agent_return_sum")),
            "agent_return_mean": _finite(reward_metrics.get("agent_return_mean")),
            "ppo_updates": int(getattr(policy, "update_count", 0)),
            "ppo_transitions": int(getattr(policy, "transition_count", 0)),
            "policy_loss": _finite(ppo.get("policy_loss")),
            "value_loss": _finite(ppo.get("value_loss")),
            "entropy": _finite(ppo.get("entropy")),
            "approx_kl": _finite(ppo.get("approx_kl")),
            "clip_fraction": _finite(ppo.get("clip_fraction")),
            "explained_variance": _finite(ppo.get("explained_variance")),
            "learning_rate": _finite(ppo.get("learning_rate")),
            "entropy_coef": _finite(ppo.get("entropy_coef")),
            "epochs_ran": int(_finite(ppo.get("epochs_ran"))),
            "early_stopped": bool(ppo.get("early_stopped", False)),
            "action_left": int(action_counts.get(0, 0)),
            "action_stop": int(action_counts.get(1, 0)),
            "action_right": int(action_counts.get(2, 0)),
            "action_switches": int(action_switches),
        }
        self.rows.append(row)

        with self.score_text_path.open("a", encoding="utf-8") as stream:
            stream.write(
                "round={round:04d} score={score:.6f} K={K:.6f} T={T:.6f} "
                "completed={completed} steps={steps} launched={red_launched} "
                "alive={red_alive} lost={red_lost} intercepted={red_intercepted} "
                "interceptors_launched={interceptors_launched} "
                "interceptors_successful={interceptors_successful} "
                "multi_kill_interceptors={interceptors_multi_kill} "
                "max_kills_per_interceptor={max_kills_by_one_interceptor} "
                "mean_train_return={agent_return_mean:.6f}\n".format(
                    **row
                )
            )
            stream.flush()

        with self.score_csv_path.open("a", encoding="utf-8", newline="") as stream:
            csv.DictWriter(stream, fieldnames=self.ROUND_FIELDS).writerow(row)
        with self.ppo_csv_path.open("a", encoding="utf-8", newline="") as stream:
            csv.DictWriter(stream, fieldnames=self.PPO_FIELDS).writerow(
                {
                    "round": round_index,
                    "update_count": row["ppo_updates"],
                    "transition_count": row["ppo_transitions"],
                    "samples": _finite(ppo.get("samples")),
                    "trajectories": _finite(ppo.get("trajectories")),
                    "rollout_environment_steps": _finite(
                        ppo.get("rollout_environment_steps")
                    ),
                    "policy_loss": row["policy_loss"],
                    "value_loss": row["value_loss"],
                    "entropy": row["entropy"],
                    "approx_kl": row["approx_kl"],
                    "clip_fraction": row["clip_fraction"],
                    "explained_variance": row["explained_variance"],
                    "learning_rate": row["learning_rate"],
                    "entropy_coef": row["entropy_coef"],
                    "epochs_ran": row["epochs_ran"],
                    "early_stopped": row["early_stopped"],
                }
            )

        record = {
            "round": int(round_index),
            "official_summary": summary,
            "training_reward": reward_metrics,
            "ppo": {
                "update_count": row["ppo_updates"],
                "transition_count": row["ppo_transitions"],
                **ppo,
            },
            "actions": {
                "left": row["action_left"],
                "stop": row["action_stop"],
                "right": row["action_right"],
                "switches": row["action_switches"],
            },
        }
        with self.metrics_jsonl_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False, default=_json_default) + "\n")
            stream.flush()

        round_json = json.dumps(record, ensure_ascii=False, indent=2, default=_json_default) + "\n"
        _atomic_text(self.result_dir / f"summary_round_{round_index:04d}.json", round_json)
        _atomic_text(self.result_dir / "latest_summary.json", round_json)
        self._write_charts()
        return row

    def _write_charts(self) -> None:
        rounds = [float(row["round"]) for row in self.rows]
        scores = [float(row["score"]) for row in self.rows]
        rolling = [
            sum(scores[max(0, index - 9): index + 1])
            / len(scores[max(0, index - 9): index + 1])
            for index in range(len(scores))
        ]
        score_svg = _single_chart_svg(
            title="R9 PPO - Official Final Score by Round",
            x_values=rounds,
            series=(("Official score", scores, "#2563eb"), ("10-round mean", rolling, "#f97316")),
            y_label="Official score (0-100)",
            fixed_y=(0.0, 100.0),
        )
        _atomic_text(self.result_dir / "round_scores.svg", score_svg)

        dashboard = _dashboard_svg(
            rounds,
            (
                (
                    "Official score",
                    (("score", scores, "#2563eb"), ("mean-10", rolling, "#f97316")),
                    (0.0, 100.0),
                ),
                (
                    "Official K / T",
                    (
                        ("K", [float(row["K"]) for row in self.rows], "#16a34a"),
                        ("T", [float(row["T"]) for row in self.rows], "#a855f7"),
                    ),
                    (0.0, 1.0),
                ),
                (
                    "Mean training return per active missile",
                    (("return", [float(row["agent_return_mean"]) for row in self.rows], "#dc2626"),),
                    None,
                ),
                (
                    "Red missiles",
                    (
                        ("launched", [float(row["red_launched"]) for row in self.rows], "#0891b2"),
                        ("alive", [float(row["red_alive"]) for row in self.rows], "#22c55e"),
                        ("lost", [float(row["red_lost"]) for row in self.rows], "#ef4444"),
                        (
                            "intercepted",
                            [float(row["red_intercepted"]) for row in self.rows],
                            "#7c3aed",
                        ),
                        (
                            "successful SAM",
                            [float(row["interceptors_successful"]) for row in self.rows],
                            "#ca8a04",
                        ),
                    ),
                    (
                        0.0,
                        max(
                            1.0,
                            1.05
                            * max(
                                (
                                    max(
                                        float(row["red_launched"]),
                                        float(row["red_alive"] + row["red_lost"]),
                                    )
                                    for row in self.rows
                                ),
                                default=1.0,
                            ),
                        ),
                    ),
                ),
            ),
        )
        _atomic_text(self.result_dir / "training_dashboard.svg", dashboard)


def _bounds(values: Sequence[float], fixed: tuple[float, float] | None) -> tuple[float, float]:
    if fixed is not None:
        return fixed
    finite_values = [value for value in values if math.isfinite(value)] or [0.0]
    low, high = min(finite_values), max(finite_values)
    if math.isclose(low, high):
        padding = max(1.0, abs(low) * 0.1)
    else:
        padding = (high - low) * 0.1
    return low - padding, high + padding


def _polyline(
    x_values: Sequence[float],
    y_values: Sequence[float],
    *,
    x0: float,
    y0: float,
    width: float,
    height: float,
    y_bounds: tuple[float, float],
    color: str,
) -> str:
    if not x_values or not y_values:
        return ""
    xmin, xmax = min(x_values), max(x_values)
    ymin, ymax = y_bounds
    xspan = max(1.0, xmax - xmin)
    yspan = max(1e-9, ymax - ymin)
    points = []
    for x_value, y_value in zip(x_values, y_values):
        x = x0 + (x_value - xmin) / xspan * width
        y = y0 + height - (y_value - ymin) / yspan * height
        points.append(f"{x:.2f},{y:.2f}")
    marker = "" if len(points) > 1 else f'<circle cx="{points[0].split(",")[0]}" cy="{points[0].split(",")[1]}" r="4" fill="{color}"/>'
    return f'<polyline points="{" ".join(points)}" fill="none" stroke="{color}" stroke-width="2.5"/>{marker}'


def _panel_svg(
    *,
    x: float,
    y: float,
    width: float,
    height: float,
    title: str,
    x_values: Sequence[float],
    series: Iterable[tuple[str, Sequence[float], str]],
    fixed_y: tuple[float, float] | None,
) -> str:
    series = tuple(series)
    all_values = [value for _, values, _ in series for value in values]
    ymin, ymax = _bounds(all_values, fixed_y)
    margin_left, margin_right, margin_top, margin_bottom = 66.0, 22.0, 42.0, 50.0
    px, py = x + margin_left, y + margin_top
    pw, ph = width - margin_left - margin_right, height - margin_top - margin_bottom
    elements = [
        f'<rect x="{x}" y="{y}" width="{width}" height="{height}" rx="8" fill="#ffffff" stroke="#d1d5db"/>',
        f'<text x="{x + 16}" y="{y + 25}" font-size="16" font-weight="600">{html.escape(title)}</text>',
    ]
    for tick in range(6):
        fraction = tick / 5.0
        ty = py + ph - fraction * ph
        value = ymin + fraction * (ymax - ymin)
        elements.append(f'<line x1="{px}" y1="{ty:.2f}" x2="{px + pw}" y2="{ty:.2f}" stroke="#e5e7eb"/>')
        elements.append(f'<text x="{px - 8}" y="{ty + 4:.2f}" text-anchor="end" font-size="11" fill="#4b5563">{value:.2f}</text>')
    elements.extend(
        [
            f'<line x1="{px}" y1="{py}" x2="{px}" y2="{py + ph}" stroke="#374151"/>',
            f'<line x1="{px}" y1="{py + ph}" x2="{px + pw}" y2="{py + ph}" stroke="#374151"/>',
        ]
    )
    for name, values, color in series:
        elements.append(
            _polyline(x_values, values, x0=px, y0=py, width=pw, height=ph, y_bounds=(ymin, ymax), color=color)
        )
    if x_values:
        elements.append(f'<text x="{px}" y="{py + ph + 22}" font-size="11">{int(min(x_values))}</text>')
        elements.append(f'<text x="{px + pw}" y="{py + ph + 22}" text-anchor="end" font-size="11">{int(max(x_values))}</text>')
    legend_x = px
    legend_y = y + height - 12
    for name, _, color in series:
        elements.append(f'<line x1="{legend_x}" y1="{legend_y}" x2="{legend_x + 18}" y2="{legend_y}" stroke="{color}" stroke-width="3"/>')
        elements.append(f'<text x="{legend_x + 24}" y="{legend_y + 4}" font-size="11">{html.escape(name)}</text>')
        legend_x += 100
    return "".join(elements)


def _single_chart_svg(
    *,
    title: str,
    x_values: Sequence[float],
    series: Iterable[tuple[str, Sequence[float], str]],
    y_label: str,
    fixed_y: tuple[float, float] | None,
) -> str:
    width, height = 1000, 600
    panel = _panel_svg(
        x=30,
        y=30,
        width=940,
        height=530,
        title=title,
        x_values=x_values,
        series=series,
        fixed_y=fixed_y,
    )
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
        '<rect width="100%" height="100%" fill="#f3f4f6"/>'
        f'<text x="16" y="300" transform="rotate(-90 16 300)" text-anchor="middle" font-size="12">{html.escape(y_label)}</text>'
        f'{panel}</svg>\n'
    )


def _dashboard_svg(
    rounds: Sequence[float],
    panels: Sequence[
        tuple[str, Iterable[tuple[str, Sequence[float], str]], tuple[float, float] | None]
    ],
) -> str:
    width, height = 1280, 900
    positions = ((30, 50), (650, 50), (30, 460), (650, 460))
    body = [
        '<rect width="100%" height="100%" fill="#f3f4f6"/>',
        '<text x="30" y="30" font-size="21" font-weight="700">R9 PPO Training Dashboard</text>',
    ]
    for (title, series, fixed_y), (x, y) in zip(panels, positions):
        body.append(
            _panel_svg(
                x=x,
                y=y,
                width=600,
                height=390,
                title=title,
                x_values=rounds,
                series=series,
                fixed_y=fixed_y,
            )
        )
    return f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">{"".join(body)}</svg>\n'
