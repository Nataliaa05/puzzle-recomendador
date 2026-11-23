import optuna

from puzzle_recsys.tuning.optuna_search import run_study, suggest

SPACE = {
    "num_leaves": {"low": 15, "high": 255, "log": True},
    "learning_rate": {"low": 0.01, "high": 0.2, "log": True},
    "feature_fraction": {"low": 0.5, "high": 1.0},
}


def test_suggest_respects_bounds_and_integer_parameters() -> None:
    study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=0))
    trial = study.ask()
    params = suggest(trial, SPACE)
    assert isinstance(params["num_leaves"], int) and 15 <= params["num_leaves"] <= 255
    assert 0.01 <= params["learning_rate"] <= 0.2
    assert 0.5 <= params["feature_fraction"] <= 1.0


def test_run_study_maximizes_and_is_reproducible() -> None:
    def objective(params: dict) -> float:
        return -((params["learning_rate"] - 0.05) ** 2)

    a = run_study(objective, SPACE, n_trials=15, seed=1)
    b = run_study(objective, SPACE, n_trials=15, seed=1)
    assert a.best_value == b.best_value
    assert abs(a.best_params["learning_rate"] - 0.05) < 0.05
    assert len(a.trials) == 15
