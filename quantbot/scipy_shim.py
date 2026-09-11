"""Normal CDF without a scipy dependency (scipy is ~100MB for one function)."""
import math


def norm_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
