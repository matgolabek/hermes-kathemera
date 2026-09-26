"""Tests for turning predictions into positions and simulating them with costs."""

from __future__ import annotations

import dataclasses
import json

import numpy as np
import pandas as pd
import pytest

import backtest.run as backtest_run
import training.run as training_run
from backtest.evaluator import calibrate_down_threshold, exit_positions, positions_from_probs, run_backtest, simulate
from backtest.metrics import calculate_metrics, periods_per_year
from config import BacktestConfig
from tests.conftest import make_raw_ohlcv

NO_COSTS = BacktestConfig(fee_rate=0.0, slippage=0.0, allow_short=False, min_confidence=0.0)

# Columns: down, flat, up
PROBS = np.array(
    [
        [0.1, 0.2, 0.7],  # up, confident
        [0.6, 0.3, 0.1],  # down, confident
        [0.2, 0.5, 0.3],  # flat
        [0.3, 0.3, 0.4],  # up, unsure
    ]
)


def test_positions_long_only():
    assert positions_from_probs(PROBS).tolist() == [1, 0, 0, 1]


def test_positions_with_shorts_and_confidence_filter():
    assert positions_from_probs(PROBS, allow_short=True).tolist() == [1, -1, 0, 1]
    assert positions_from_probs(PROBS, allow_short=True, min_confidence=0.5).tolist() == [1, -1, 0, 0]


def test_exit_on_down_stays_long_unless_down_is_likely():
    assert positions_from_probs(PROBS, rule="exit-on-down", down_prob_threshold=0.5).tolist() == [1, 0, 1, 1]
    assert positions_from_probs(PROBS, rule="exit-on-down", down_prob_threshold=0.25).tolist() == [1, 0, 1, 0]
    shorts = positions_from_probs(PROBS, rule="exit-on-down", down_prob_threshold=0.5, allow_short=True)
    assert shorts.tolist() == [1, -1, 1, 1]


def test_binary_probabilities():
    probs = np.array([[0.7, 0.3], [0.2, 0.8], [0.45, 0.55]])  # down, rest

    assert positions_from_probs(probs, class_names=["down", "rest"]).tolist() == [0, 1, 1]
    exits = positions_from_probs(probs, class_names=["down", "rest"], rule="exit-on-down", down_prob_threshold=0.4)
    assert exits.tolist() == [0, 1, 0]


def test_unknown_rule_rejected():
    with pytest.raises(ValueError, match="rule"):
        positions_from_probs(PROBS, rule="hodl")


def test_calibrated_threshold_exits_for_the_requested_share():
    rng = np.random.default_rng(0)
    p_down = rng.uniform(0.0, 0.3, 10_000)
    probs = np.column_stack([p_down, 1 - p_down])

    threshold = calibrate_down_threshold(probs, ["down", "rest"], exit_share=0.1)
    positions = positions_from_probs(probs, class_names=["down", "rest"], rule="exit-on-down", down_prob_threshold=threshold)

    assert threshold == pytest.approx(0.27, abs=0.01)
    assert (positions == 0).mean() == pytest.approx(0.1, abs=0.002)


def test_calibration_rejects_bad_input():
    probs = np.array([[0.2, 0.8]])
    with pytest.raises(ValueError):
        calibrate_down_threshold(probs, ["down", "rest"], exit_share=1.5)
    with pytest.raises(ValueError):
        calibrate_down_threshold(np.empty((0, 2)), ["down", "rest"], exit_share=0.1)


def test_hysteresis_waits_for_the_reentry_level():
    scores = np.array([0.1, 0.5, 0.3, 0.25, 0.1, 0.3, 0.45, 0.1])

    plain = exit_positions(scores, exit_threshold=0.4)
    sticky = exit_positions(scores, exit_threshold=0.4, reentry_threshold=0.2)

    assert plain.tolist() == [1, 0, 1, 1, 1, 1, 0, 1]
    assert sticky.tolist() == [1, 0, 0, 0, 1, 1, 0, 1]
    assert exit_positions(scores, 0.4, 0.2, allow_short=True).tolist() == [1, -1, -1, -1, 1, 1, -1, 1]


def test_hysteresis_cuts_trades_on_clustered_scores():
    # Like volatility: a slowly drifting level plus hour-to-hour jitter around it.
    rng = np.random.default_rng(0)
    level = np.zeros(5000)
    for i in range(1, len(level)):
        level[i] = 0.98 * level[i - 1] + rng.normal(0, 0.01)
    scores = 0.3 + level + rng.normal(0, 0.02, len(level))
    exit_at, reentry_at = np.quantile(scores, 0.8), np.quantile(scores, 0.5)

    plain = exit_positions(scores, exit_threshold=exit_at)
    sticky = exit_positions(scores, exit_threshold=exit_at, reentry_threshold=reentry_at)

    assert np.count_nonzero(np.diff(sticky)) < np.count_nonzero(np.diff(plain)) / 3


def test_reentry_above_exit_rejected():
    with pytest.raises(ValueError):
        exit_positions(np.array([0.1]), exit_threshold=0.2, reentry_threshold=0.3)


def test_strategy_config_validates_reentry_share(monkeypatch):
    import argparse

    parser = argparse.ArgumentParser()
    backtest_run.add_strategy_args(parser)

    cfg = backtest_run.strategy_config(parser.parse_args(["--exit-share", "0.2", "--reentry-share", "0.5"]))
    assert (cfg.rule, cfg.exit_share, cfg.reentry_share) == ("exit-on-down", 0.2, 0.5)
    with pytest.raises(SystemExit):
        backtest_run.strategy_config(parser.parse_args(["--exit-share", "0.2", "--reentry-share", "0.1"]))
    with pytest.raises(SystemExit):
        backtest_run.strategy_config(parser.parse_args(["--reentry-share", "0.5"]))


def test_volatility_rule_exits_in_volatile_periods():
    from data_pipeline.pipeline import prepare_datasets

    data = prepare_datasets(make_raw_ohlcv(1500), 16, 0.7, 0.15, timeframe="1h")
    cfg = dataclasses.replace(NO_COSTS, exit_share=0.2)

    result = backtest_run.volatility_rule(data.val, data.train, data.feature_columns, cfg, periods=8760)

    vol = data.val.X[:, -1, data.feature_columns.index("volatility_24")]
    out = result.frame["position"].to_numpy() == 0
    assert out.any() and vol[out].min() > vol[~out].max()
    assert backtest_run.volatility_rule(data.val, data.train, data.feature_columns, NO_COSTS, 8760) is None


def test_run_tag_names_strategy_settings():
    base = dataclasses.replace(NO_COSTS, rule="argmax")
    assert backtest_run.run_tag(base) == "argmax"
    assert backtest_run.run_tag(dataclasses.replace(base, min_confidence=0.5, allow_short=True)) == "argmax_conf0.5_short"
    assert backtest_run.run_tag(dataclasses.replace(base, rule="exit-on-down", down_prob_threshold=0.4)) == "exit_p0.4"
    assert backtest_run.run_tag(dataclasses.replace(base, exit_share=0.1)) == "exit_top10pct"
    assert backtest_run.run_tag(dataclasses.replace(base, exit_share=0.2, reentry_share=0.5)) == "exit_top20pct_re50pct"


def test_simulate_charges_costs_on_every_position_change():
    positions = np.array([1, 1, 0, -1, 1])
    log_returns = np.log([1.01, 1.02, 1.03, 0.98, 1.00])

    frame = simulate(positions, log_returns, cost_per_trade=0.001)

    # enter, hold, exit, open short, flip to long (2 units) and final close
    assert frame["turnover"].tolist() == [1, 0, 1, 1, 3]
    assert frame["cost"].tolist() == pytest.approx([0.001, 0, 0.001, 0.001, 0.003])
    assert frame["gross_return"].tolist() == pytest.approx([0.01, 0.02, 0.0, 0.02, 0.0])
    assert frame["equity"].iloc[-1] == pytest.approx(np.prod(1 + frame["net_return"]))


def test_buy_and_hold_without_costs_tracks_price():
    close = np.array([100.0, 110.0, 99.0, 120.0])
    log_returns = np.diff(np.log(close))

    result = run_backtest(np.ones(3), log_returns, NO_COSTS, periods=8760)

    assert result.metrics.total_return == pytest.approx(0.2)
    assert result.metrics.max_drawdown == pytest.approx(-0.1)
    assert result.metrics.exposure == 1.0
    assert result.metrics.trades == 2  # buy at the start, sell at the end


def test_costs_separate_net_from_gross():
    log_returns = np.zeros(4)
    config = dataclasses.replace(NO_COSTS, fee_rate=0.001, slippage=0.0005)

    result = run_backtest(np.array([1, 0, 1, 0]), log_returns, config, periods=8760)

    assert result.gross_metrics.total_return == pytest.approx(0.0)
    assert result.metrics.total_return < 0
    assert result.metrics.costs == pytest.approx(4 * 0.0015)
    assert result.metrics.trades == 4


def test_metrics_edge_cases():
    flat = calculate_metrics(np.zeros(5), np.zeros(5), np.zeros(5), trades=0, periods=8760)

    assert flat.sharpe == 0.0 and flat.hit_rate == 0.0 and flat.max_drawdown == 0.0
    with pytest.raises(ValueError):
        calculate_metrics(np.array([]), np.array([]), np.array([]), trades=0, periods=8760)


def test_drawdown_counts_losses_from_starting_capital():
    metrics = calculate_metrics(np.array([-0.1, 0.05]), np.ones(2), np.zeros(2), trades=2, periods=8760)

    assert metrics.max_drawdown == pytest.approx(-0.1)


def test_periods_per_year():
    assert periods_per_year("1h") == 8760
    assert periods_per_year("1d") == 365


def train_small(tmp_path, monkeypatch, raw, label_mode="three_class"):
    monkeypatch.setattr(training_run.CCXTLoader, "load", lambda self: raw)
    monkeypatch.setenv("LABEL_MODE", label_mode)
    monkeypatch.setenv("CHECKPOINT_DIR", str(tmp_path))
    monkeypatch.setenv("EPOCHS", "1")
    monkeypatch.setenv("SEQUENCE_LENGTH", "16")
    training_run.main(["--models", "linear"])


def test_cli_backtests_trained_models(tmp_path, monkeypatch, capsys):
    raw = make_raw_ohlcv(1500)
    train_small(tmp_path, monkeypatch, raw)
    monkeypatch.setattr(backtest_run.CCXTLoader, "load", lambda self: raw)

    results = backtest_run.main(["--allow-short"])

    out = capsys.readouterr().out
    assert "buy&hold" in out and "long/short" in out
    assert list(results) == ["buy&hold", "linear"]
    summary = json.loads((tmp_path / "backtest_val_argmax_short.json").read_text())
    assert summary["config"]["allow_short"] is True
    equity = pd.read_csv(tmp_path / "backtest_val_argmax_short_equity.csv", index_col=0)
    assert list(equity.columns) == ["buy&hold", "linear"]
    assert len(equity) == len(results["linear"].frame)


def test_cli_refuses_to_trade_on_training_period(tmp_path, monkeypatch):
    raw = make_raw_ohlcv(3000)
    train_small(tmp_path, monkeypatch, raw)
    # Less data now: the validation split falls inside the period the model was trained on.
    monkeypatch.setattr(backtest_run.CCXTLoader, "load", lambda self: raw.iloc[:1500])

    with pytest.raises(ValueError, match="retrain"):
        backtest_run.main([])


def test_cli_without_checkpoints_exits(tmp_path, monkeypatch):
    monkeypatch.setenv("CHECKPOINT_DIR", str(tmp_path))

    with pytest.raises(SystemExit):
        backtest_run.main([])


def test_cli_exit_on_down_with_binary_model(tmp_path, monkeypatch, capsys):
    raw = make_raw_ohlcv(1500)
    train_small(tmp_path, monkeypatch, raw, label_mode="binary")
    monkeypatch.setattr(backtest_run.CCXTLoader, "load", lambda self: raw)

    results = backtest_run.main(["--rule", "exit-on-down", "--down-threshold", "0.0"])

    assert "exit-on-down" in capsys.readouterr().out
    # Threshold 0 exits whenever P(down) > 0, i.e. always: never in the market.
    assert results["linear"].metrics.exposure == 0.0
    summary = json.loads((tmp_path / "backtest_val_exit_p0.json").read_text())
    assert summary["config"]["rule"] == "exit-on-down"
    assert summary["results"]["linear"]["down_prob_threshold"] == 0.0


def test_walk_forward_cli(tmp_path, monkeypatch, capsys):
    import backtest.walk_forward as walk_forward

    monkeypatch.setattr(walk_forward.CCXTLoader, "load", lambda self: make_raw_ohlcv(2500))
    monkeypatch.setenv("CHECKPOINT_DIR", str(tmp_path))
    monkeypatch.setenv("EPOCHS", "1")
    monkeypatch.setenv("SEQUENCE_LENGTH", "16")

    report = walk_forward.main(["--models", "linear", "--folds", "3", "--rule", "exit-on-down"])

    out = capsys.readouterr().out
    assert "stitched" in out
    assert [f["fold"] for f in report["folds"]] == [1, 2, 3]
    assert set(report["totals"]) == {"buy&hold", "linear"}
    assert 0 <= report["totals"]["linear"]["folds_beating_buy_and_hold"] <= 3
    periods = [f["eval_period"] for f in report["folds"]]
    assert all(a[1] < b[0] for a, b in zip(periods, periods[1:]))
    out_dir = tmp_path / "walk_forward" / "three_class_h1_exit_p0.5"
    equity = pd.read_csv(out_dir / "equity.csv", index_col=0)
    assert list(equity.columns) == ["buy&hold", "linear"]
    assert (out_dir / "fold_3" / "linear.pt").exists()


def test_cli_exit_share_calibrates_before_the_traded_period(tmp_path, monkeypatch, capsys):
    raw = make_raw_ohlcv(3000)
    train_small(tmp_path, monkeypatch, raw, label_mode="binary")
    monkeypatch.setattr(backtest_run.CCXTLoader, "load", lambda self: raw)
    calibrations = []
    original = backtest_run.score_threshold

    def spy(scores, share):
        calibrations.append(len(scores))
        return original(scores, share)

    monkeypatch.setattr(backtest_run, "score_threshold", spy)

    results = backtest_run.main(["--exit-share", "0.2"])

    assert "top 20%" in capsys.readouterr().out
    linear = results["linear"]
    assert linear.down_prob_threshold is not None
    # Model and vol_rule both calibrate on as many pre-validation windows as the validation split has.
    assert calibrations == [len(linear.frame)] * 2
    # Roughly the requested share is spent out of the market (the distribution can drift).
    assert 0.5 < linear.metrics.exposure < 0.97
    summary = json.loads((tmp_path / "backtest_val_exit_top20pct.json").read_text())
    assert summary["config"]["exit_share"] == 0.2
    assert summary["config"]["rule"] == "exit-on-down"


def test_calibration_split_precedes_the_traded_split():
    from data_pipeline.pipeline import prepare_datasets

    data = prepare_datasets(make_raw_ohlcv(1500), 16, 0.7, 0.15, timeframe="1h")

    before_val = backtest_run.calibration_split(data, "val")
    before_test = backtest_run.calibration_split(data, "test")

    assert len(before_val) == len(data.val)
    assert before_val.timestamps[-1] == data.train.timestamps[-1]
    assert before_test is data.val


def test_walk_forward_exit_share_calibrates_each_fold(tmp_path, monkeypatch):
    import backtest.walk_forward as walk_forward

    monkeypatch.setattr(walk_forward.CCXTLoader, "load", lambda self: make_raw_ohlcv(2500))
    monkeypatch.setenv("CHECKPOINT_DIR", str(tmp_path))
    monkeypatch.setenv("EPOCHS", "1")
    monkeypatch.setenv("SEQUENCE_LENGTH", "16")
    monkeypatch.setenv("LABEL_MODE", "binary")

    report = walk_forward.main(["--models", "linear", "--folds", "2", "--exit-share", "0.1"])

    thresholds = [f["models"]["linear"]["down_prob_threshold"] for f in report["folds"]]
    assert all(t is not None for t in thresholds)
    assert report["config"]["exit_share"] == 0.1
    assert (tmp_path / "walk_forward" / "binary_h1_exit_top10pct" / "walk_forward.json").exists()


def test_walk_forward_with_hysteresis_includes_volatility_baseline(tmp_path, monkeypatch, capsys):
    import backtest.walk_forward as walk_forward

    monkeypatch.setattr(walk_forward.CCXTLoader, "load", lambda self: make_raw_ohlcv(2500))
    monkeypatch.setenv("CHECKPOINT_DIR", str(tmp_path))
    monkeypatch.setenv("EPOCHS", "1")
    monkeypatch.setenv("SEQUENCE_LENGTH", "16")
    monkeypatch.setenv("LABEL_MODE", "binary")

    report = walk_forward.main(["--models", "linear", "--folds", "2", "--exit-share", "0.2", "--reentry-share", "0.5"])

    assert "vol_rule" in capsys.readouterr().out
    assert list(report["totals"]) == ["buy&hold", "vol_rule", "linear"]
    fold = report["folds"][0]["models"]["linear"]
    assert fold["reentry_threshold"] < fold["down_prob_threshold"]
    assert (tmp_path / "walk_forward" / "binary_h1_exit_top20pct_re50pct" / "equity.csv").exists()
