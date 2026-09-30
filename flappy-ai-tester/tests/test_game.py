"""Test automatici del gioco guidati dalla percezione visiva.

Girano sul clone locale in lockstep, quindi sono deterministici e veloci:
    pytest -q
"""
from __future__ import annotations

import pytest

from flappy_ai.agents import HeuristicAgent
from flappy_ai.browser import clone_url
from flappy_ai.env import FlappyEnv
from flappy_ai.vision import PROFILES


@pytest.fixture
def env():
    e = FlappyEnv(clone_url(seed=7, lockstep=True), PROFILES["clone"], lockstep=True, max_score=10)
    yield e
    e.close()


def play(env, policy, max_steps=5000):
    obs, _ = env.reset()
    for _ in range(max_steps):
        obs, _, done, truncated, _ = env.step(policy(obs))
        if done or truncated:
            return done, truncated
    return False, False


def test_game_loads_and_bird_is_visible(env):
    env.reset()
    assert env.frame.bird is not None, "uccello non rilevato nel canvas"
    assert env.frame.ground_strip is not None


def test_flap_moves_bird_up(env):
    env.reset()
    for _ in range(15):                       # lascia cadere l'uccello
        env.step(0)
    y_before = env.frame.bird_center[1]
    env.step(1)
    env.step(0)
    assert env.frame.bird_center[1] < y_before - 3, "l'input non fa salire l'uccello"


def test_pipes_appear_and_scroll_left(env):
    agent = HeuristicAgent()
    obs, _ = env.reset()
    for _ in range(60):
        obs, *_ = env.step(agent.act(obs))
    assert env.frame.pipes, "nessun tubo rilevato"
    x_before = min(p.x0 for p in env.frame.pipes)
    obs, *_ = env.step(agent.act(obs))
    assert min(p.x0 for p in env.frame.pipes) < x_before


def test_idle_bird_dies_on_ground(env):
    done, _ = play(env, lambda obs: 0)
    assert done, "game over non rilevato"
    assert env.stats.death_cause == "ground"
    assert env.browser.ground_truth()["deathCause"] == "ground"


def test_score_increases_and_matches_ground_truth(env):
    agent = HeuristicAgent()
    done, truncated = play(env, agent.act)
    assert env.stats.score > 0, "il punteggio non aumenta superando i tubi"
    assert env.stats.score == env.browser.ground_truth()["score"], "punteggio visivo diverso da quello reale"


def test_game_restarts_after_game_over(env):
    play(env, lambda obs: 0)
    env.reset()
    assert env.browser.ground_truth()["mode"] == "playing"
    assert env.stats.score == 0
