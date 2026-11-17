"""SupeRR Selector Round 1. Run locally against the organizer's original ZIP.

Requires Python 3.10+, numpy, pandas and scikit-learn.
Example: python rr_pressure_model.py data_pack.zip --out rr_results
Outputs player rankings, sensitivity results, match evidence and diagnostics.
"""
import argparse
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import GroupKFold

WICKET_COST = 5
SHRINK = 60
MIN_HIGH = 36
MIN_ORDINARY = 72
MIN_MATCHES = 8
BOOTSTRAPS = 500
CREDITED = {"Caught", "Bowled", "LBW", "Stump", "Hit Wicket"}


def load_deliveries(path):
    with ZipFile(path) as source:
        def read(name):
            return pd.read_csv(source.open(f"Data Pack/{name}.csv"), low_memory=False)

        balls, matches, comps = (read(name) for name in ("ball_by_ball", "matches", "competitions"))
    audit = {"raw_deliveries": len(balls), "match_metadata": len(matches),
             "matches_with_deliveries": balls.match_id.nunique()}
    matches = matches.merge(comps[["comp_id", "comp_name", "match_format"]],
                            on="comp_id", validate="many_to_one")
    full = matches[["first_inn_revised_overs", "second_inn_revised_overs", "revised_target"]].isna().all(axis=1)
    matches = matches[full & matches.win_team_id.notna()].copy()
    matches["circuit"] = np.select(
        [matches.comp_name.str.match(r"IPL\b"), matches.match_format.eq("T20I")],
        ["IPL", "T20I"], default="Emerging"
    )
    balls = balls[balls.innings.isin([1, 2])].merge(
        matches[["match_id", "match_date", "win_team_id", "circuit"]],
        on="match_id", validate="many_to_one"
    ).sort_values(["match_id", "innings", "innings_ball_num"]).reset_index(drop=True)
    key = ["match_id", "innings"]
    assert not balls.duplicated(key + ["innings_ball_num"]).any()
    balls["year"] = pd.to_datetime(balls.match_date).dt.year
    balls["wide"] = balls.extras_type.eq("wide")
    balls["noball"] = balls.extras_type.isin(["Overstep", "other"])
    balls["penalty"] = balls.extras_type.eq("Penalty")
    balls["legal"] = ~(balls.wide | balls.noball | balls.penalty)
    balls["team_runs"] = balls.runs + balls.extras1_runs + balls.extras_2_runs
    balls["team_wicket"] = balls.is_wicket.astype(int)
    groups = balls.groupby(key, sort=False)
    balls["pre_runs"] = groups.team_runs.cumsum() - balls.team_runs
    balls["pre_wickets"] = groups.team_wicket.cumsum() - balls.team_wicket
    balls["used"] = groups.legal.cumsum() - balls.legal
    balls["left"] = 120 - balls.used
    targets = balls[balls.innings.eq(1)].groupby("match_id").team_runs.sum() + 1
    balls["needed"] = balls.match_id.map(targets) - balls.pre_runs
    # Keep matches with two coherent innings and remove invalid terminal states.
    valid = balls.groupby(key).legal.sum().unstack()
    valid = valid.dropna().index[valid.dropna().le(120).all(axis=1)]
    balls = balls[balls.match_id.isin(valid) & balls.left.gt(0) & balls.pre_wickets.lt(10)
                  & (balls.innings.eq(1) | balls.needed.gt(0))].copy()
    balls["win"] = balls.batting_team_id.eq(balls.win_team_id).astype(int)
    balls["phase"] = pd.cut(balls.used, [-1, 35, 95, 119], labels=["PP", "Middle", "Death"])
    balls["wicket_band"] = (balls.pre_wickets // 2).clip(upper=3)
    balls["bat_role"] = pd.cut(balls.batting_order, [0, 3, 5, 7, 20],
                               labels=["1-3", "4-5", "6-7", "8+"])
    balls["bowl_role"] = balls.pace_or_spin.fillna("Unknown")
    audit.update({"model_deliveries": len(balls), "model_matches": balls.match_id.nunique(),
                  "circuit_matches": balls.groupby("circuit").match_id.nunique().to_dict(),
                  "ambiguous_other": int(balls.extras_type.eq("other").sum())})
    return balls.reset_index(drop=True), audit


def chase_features(frame, run_delta=0, wicket_delta=0):
    left = frame.left.to_numpy()
    wickets = np.minimum(frame.pre_wickets.to_numpy() + wicket_delta, 10)
    needed = np.maximum(frame.needed.to_numpy() - run_delta, 0)
    return np.column_stack([
        np.log1p(left), wickets, wickets ** 2 / 10, needed / 30,
        np.log1p(6 * needed / left), wickets * np.log1p(left) / 5,
        frame.circuit.eq("IPL"), frame.circuit.eq("Emerging")
    ])


def pressure_states(train, test):
    state = test.copy()
    chase_train = train[train.innings.eq(2) & train.legal]
    chase_test = state[state.innings.eq(2)]
    chase_model = LogisticRegression(max_iter=800)
    chase_model.fit(chase_features(chase_train), chase_train.win)
    predict = lambda **kw: chase_model.predict_proba(chase_features(chase_test, **kw))[:, 1]
    probability = predict()
    plus_run = np.where(chase_test.needed.eq(1), 1, predict(run_delta=1))
    minus_wicket = np.where(chase_test.pre_wickets.eq(9), 0, predict(wicket_delta=1))
    swing = np.maximum(plus_run - probability, 0) + np.maximum(probability - minus_wicket, 0)
    state["p_win"] = np.nan
    state.loc[chase_test.index, "p_win"] = probability
    state["swing"] = np.nan
    state.loc[chase_test.index, "swing"] = swing
    first_train = train[train.innings.eq(1) & train.legal]
    par = first_train.groupby(["circuit", "used"]).pre_runs.median()
    fallback = first_train.groupby("used").pre_runs.median()
    expected = par.reindex(pd.MultiIndex.from_frame(state[["circuit", "used"]])).to_numpy()
    expected = np.where(np.isnan(expected), state.used.map(fallback), expected)
    state["lag"] = expected - state.pre_runs
    return state


def label_pressure(state, strict=False):
    phase_threshold = np.select([state.phase.eq("PP"), state.phase.eq("Middle")],
                                [3, 4], default=5)
    first_high = state.used.ge(12) & state.pre_wickets.ge(phase_threshold) & state.lag.ge(10 if strict else 6)
    chase_high = state.p_win.between(.25 if strict else .15, .75 if strict else .85)
    chase_high &= state.swing.ge(.20 if strict else .15)
    return np.where(state.innings.eq(1), first_high, chase_high)


def outcomes(state, side, wicket_cost):
    result = state.copy()
    if side == "bat":
        result = result[~result.wide & ~result.penalty].copy()
        result["player"] = result.striker_id
        result["name"] = result.striker_name
        result["runs_value"] = result.runs
        result["wicket_value"] = (result.is_wicket & result.dismissed_player_id.eq(result.striker_id)).astype(int)
        result["role"] = result.bat_role
    else:
        result = result[~result.penalty].copy()
        result["player"] = result.bowler_id
        result["name"] = result.bowler_name
        result["runs_value"] = -(result.runs + np.where(result.wide | result.noball, result.extras1_runs, 0))
        result["wicket_value"] = result.dismissal_type.isin(CREDITED).astype(int)
        result["role"] = result.bowl_role
    result["value"] = result.runs_value + (wicket_cost if side == "bowl" else -wicket_cost) * result.wicket_value
    return result


def residuals(train, test, side, wicket_cost):
    train = outcomes(train, side, wicket_cost)
    test = outcomes(test, side, wicket_cost)
    # Refine a global mean with role- and state-matched training observations.
    expected = np.full(len(test), train.value.mean())
    keys = ["circuit", "innings", "phase", "high", "wicket_band", "role"]
    for size in (3, 4, 5, 6):
        columns = keys[:size]
        stats = train.groupby(columns, observed=True).value.agg(["sum", "size"])
        cells = stats.reindex(pd.MultiIndex.from_frame(test[columns])).fillna(0)
        expected = (cells["sum"].to_numpy() + 40 * expected) / (cells["size"].to_numpy() + 40)
    test["residual"] = test.value - expected
    return test


def crossfit(balls, specs):
    data = {name: {"bat": [], "bowl": []} for name in specs}
    prediction_rows = []
    splitter = GroupKFold(n_splits=5)
    for fold, (train_idx, test_idx) in enumerate(splitter.split(balls, groups=balls.match_id)):
        train, test = balls.iloc[train_idx], balls.iloc[test_idx]
        # An inner match split supplies out-of-sample state probabilities to the peer baseline.
        peer_states = []
        inner = GroupKFold(n_splits=4)
        for fit_idx, peer_idx in inner.split(train, groups=train.match_id):
            peer_states.append(pressure_states(train.iloc[fit_idx], train.iloc[peer_idx]))
        peer = pd.concat(peer_states)
        held = pressure_states(train, test)
        prediction_rows.append(held[held.innings.eq(2) & held.legal][["win", "p_win", "year"]])
        for name, (strict, wicket_cost) in specs.items():
            peer["high"] = label_pressure(peer, strict)
            held["high"] = label_pressure(held, strict)
            for side in ("bat", "bowl"):
                result = residuals(peer, held, side, wicket_cost)
                data[name][side].append(result)
    data = {name: {side: pd.concat(parts) for side, parts in sides.items()} for name, sides in data.items()}
    return data, pd.concat(prediction_rows)


def score(high_sum, high_n, base_sum, base_n):
    return high_sum / (high_n + SHRINK) - .35 * base_sum / (base_n + SHRINK)


def rank(frame, min_high=MIN_HIGH, min_ordinary=MIN_ORDINARY, min_matches=MIN_MATCHES, bootstrap=False):
    counts = frame.groupby(["player", "high"]).residual.agg(["size", "sum"]).unstack(fill_value=0)
    high = frame[frame.high]
    matches = high.groupby("player").match_id.nunique().rename("matches")
    ranked = pd.DataFrame({
        "high_n": counts["size"][True], "base_n": counts["size"][False],
        "high_sum": counts["sum"][True], "base_sum": counts["sum"][False],
    }).join(matches).fillna(0)
    ranked = ranked.query("high_n >= @min_high and base_n >= @min_ordinary and matches >= @min_matches").copy()
    ranked["high_peer"] = 100 * ranked.high_sum / ranked.high_n
    ranked["ordinary_peer"] = 100 * ranked.base_sum / ranked.base_n
    ranked["uplift"] = ranked.high_peer - ranked.ordinary_peer
    ranked["score"] = 100 * score(ranked.high_sum, ranked.high_n, ranked.base_sum, ranked.base_n)
    names = frame.groupby(["player", "name"]).size().sort_values(ascending=False, kind="stable")
    names = names[~names.index.get_level_values("player").duplicated()].reset_index().set_index("player")["name"]
    ranked = ranked.join(names)
    by_match = high.groupby(["player", "match_id"]).residual.sum()
    ranked["positive_matches"] = by_match.gt(0).groupby("player").sum()
    dominant = high.groupby(["player", "phase"], observed=True).size().unstack(fill_value=0)
    ranked["top_phase"] = dominant.idxmax(axis=1)
    ranked["top_phase_n"] = dominant.max(axis=1)
    ranked["ipl_n"] = high[high.circuit.eq("IPL")].groupby("player").size().reindex(ranked.index, fill_value=0)
    if bootstrap:
        rng = np.random.default_rng(202711)
        ids = np.sort(frame.match_id.unique())
        weights = rng.multinomial(len(ids), np.full(len(ids), 1 / len(ids)), size=BOOTSTRAPS)
        grouped = frame.groupby(["player", "high", "match_id"]).residual.agg(["sum", "size"])
        arrays = []
        for flag in (True, False):
            matrix = grouped.xs(flag, level="high")
            for metric in ("sum", "size"):
                pivot = matrix[metric].unstack(fill_value=0).reindex(index=ranked.index, columns=ids, fill_value=0)
                arrays.append(weights @ pivot.to_numpy().T)
        samples = 100 * score(*arrays)
        ranked["low"], ranked["high"] = np.percentile(samples, [5, 95], axis=0)
        top = np.argsort(-samples, axis=1, kind="stable")[:, :10]
        ranked["top10_pct"] = 100 * np.bincount(top.ravel(), minlength=len(ranked)) / BOOTSTRAPS
    ranked = ranked.sort_values(["score", "high_n"], ascending=False, kind="stable")
    ranked.insert(0, "rank", np.arange(1, len(ranked) + 1))
    return ranked


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("zip_path", type=Path)
    parser.add_argument("--out", type=Path, default=Path("rr_results"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    balls, audit = load_deliveries(args.zip_path)
    specs = {"primary": (False, WICKET_COST), "strict": (True, WICKET_COST),
             "wicket3": (False, 3), "wicket8": (False, 8)}
    data, probability = crossfit(balls, specs)
    chase = probability
    audit["chase_auc"] = roc_auc_score(chase.win, chase.p_win)
    audit["chase_brier"] = brier_score_loss(chase.win, chase.p_win)
    audit["chase_brier_base"] = brier_score_loss(chase.win, np.full(len(chase), chase.win.mean()))
    # Evaluate a separate, genuinely forward season with all references built before 2026.
    earlier, future = balls[balls.year.lt(2026)], balls[balls.year.eq(2026)]
    forward_peer = []
    inner = GroupKFold(n_splits=5)
    for fit_idx, peer_idx in inner.split(earlier, groups=earlier.match_id):
        forward_peer.append(pressure_states(earlier.iloc[fit_idx], earlier.iloc[peer_idx]))
    earlier_state = pd.concat(forward_peer)
    future_state = pressure_states(earlier, future)
    for state in (earlier_state, future_state):
        state["high"] = label_pressure(state)
    future_chase = future_state[future_state.innings.eq(2) & future_state.legal]
    audit["forward_chase_auc"] = roc_auc_score(future_chase.win, future_chase.p_win)
    audit["forward_chase_brier"] = brier_score_loss(future_chase.win, future_chase.p_win)
    audit["forward_chase_brier_base"] = brier_score_loss(
        future_chase.win, np.full(len(future_chase), earlier.query("innings == 2 and legal").win.mean())
    )
    for side, frame in data["primary"].items():
        primary = rank(frame, bootstrap=True)
        variants = {
            "ipl": rank(frame[frame.circuit.eq("IPL")], 24, 48, 5),
            "strict": rank(data["strict"][side], 24, 72, 6),
            "wicket3": rank(data["wicket3"][side]),
            "wicket8": rank(data["wicket8"][side]),
            "pre2026": rank(frame[frame.year.lt(2026)]),
            "forward2026": rank(residuals(earlier_state, future_state, side, WICKET_COST), 12, 24, 3),
        }
        for variant, result in variants.items():
            primary[f"{variant}_rank"] = result["rank"]
            primary[f"{variant}_score"] = result["score"]
        primary.to_csv(args.out / f"{side}_rankings.csv", float_format="%.5f")
        candidates = primary.head(12).index
        evidence = frame[frame.player.isin(candidates) & frame.high].groupby(
            ["player", "phase", "innings"], observed=True
        ).agg(balls=("residual", "size"), impact=("residual", "mean"))
        evidence.to_csv(args.out / f"{side}_roles.csv", float_format="%.5f")
        audit[side] = {
            "eligible": len(primary), "events": len(frame), "pressure": int(frame.high.sum()),
            "pressure_first": int((frame.high & frame.innings.eq(1)).sum()),
            "pressure_chase": int((frame.high & frame.innings.eq(2)).sum()),
            "overlap": {key: len(set(primary.head(10).index) & set(result.head(10).index))
                        for key, result in variants.items()}
        }
    (args.out / "diagnostics.json").write_text(json.dumps(audit, indent=2) + "\n")


if __name__ == "__main__":
    main()
