import pytest

from quantbot.config import Config
from quantbot.data.synthetic import generate


@pytest.fixture(scope="session")
def btc():
    return generate("BTC/USDT", bars=3000, seed=1)


@pytest.fixture(scope="session")
def eth():
    return generate("ETH/USDT", bars=3000, seed=1, start_price=1900.0)


@pytest.fixture
def cfg():
    c = Config()
    c.data.source = "synthetic"
    c.model.train_bars = 900
    c.model.min_train_bars = 400
    c.model.retrain_every = 400
    c.model.n_seeds = 1
    c.model.max_iter = 40
    c.alerts.channels = ["console"]
    return c
