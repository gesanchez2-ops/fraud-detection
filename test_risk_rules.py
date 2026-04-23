from risk_rules import label_risk, score_transaction


def _base_tx(**overrides):
    """Minimal zero-risk transaction; override individual fields per test."""
    tx = {
        "device_risk_score": 5,
        "is_international": 0,
        "amount_usd": 50.0,
        "velocity_24h": 1,
        "failed_logins_24h": 0,
        "prior_chargebacks": 0,
    }
    tx.update(overrides)
    return tx


# ---------------------------------------------------------------------------
# label_risk
# ---------------------------------------------------------------------------

def test_label_risk_thresholds():
    assert label_risk(10) == "low"
    assert label_risk(35) == "medium"
    assert label_risk(75) == "high"


def test_label_risk_boundaries():
    assert label_risk(29) == "low"
    assert label_risk(30) == "medium"
    assert label_risk(59) == "medium"
    assert label_risk(60) == "high"


# ---------------------------------------------------------------------------
# score_transaction — baseline
# ---------------------------------------------------------------------------

def test_clean_transaction_scores_zero():
    assert score_transaction(_base_tx()) == 0


# ---------------------------------------------------------------------------
# score_transaction — amount
# ---------------------------------------------------------------------------

def test_large_amount_adds_risk():
    assert score_transaction(_base_tx(amount_usd=1200)) == 25


def test_moderate_amount_adds_10():
    assert score_transaction(_base_tx(amount_usd=750)) == 10


def test_small_amount_adds_nothing():
    assert score_transaction(_base_tx(amount_usd=499)) == 0


# ---------------------------------------------------------------------------
# score_transaction — device risk
# ---------------------------------------------------------------------------

def test_high_device_risk_adds_25():
    assert score_transaction(_base_tx(device_risk_score=70)) == 25


def test_medium_device_risk_adds_10():
    assert score_transaction(_base_tx(device_risk_score=40)) == 10


def test_low_device_risk_adds_nothing():
    assert score_transaction(_base_tx(device_risk_score=39)) == 0


# ---------------------------------------------------------------------------
# score_transaction — international flag
# ---------------------------------------------------------------------------

def test_international_adds_15():
    assert score_transaction(_base_tx(is_international=1)) == 15


def test_domestic_adds_nothing():
    assert score_transaction(_base_tx(is_international=0)) == 0


# ---------------------------------------------------------------------------
# score_transaction — velocity
# ---------------------------------------------------------------------------

def test_high_velocity_adds_20():
    assert score_transaction(_base_tx(velocity_24h=6)) == 20


def test_medium_velocity_adds_5():
    assert score_transaction(_base_tx(velocity_24h=3)) == 5


def test_low_velocity_adds_nothing():
    assert score_transaction(_base_tx(velocity_24h=2)) == 0


# ---------------------------------------------------------------------------
# score_transaction — failed logins
# ---------------------------------------------------------------------------

def test_high_failed_logins_adds_20():
    assert score_transaction(_base_tx(failed_logins_24h=5)) == 20


def test_medium_failed_logins_adds_10():
    assert score_transaction(_base_tx(failed_logins_24h=2)) == 10


def test_no_failed_logins_adds_nothing():
    assert score_transaction(_base_tx(failed_logins_24h=0)) == 0


# ---------------------------------------------------------------------------
# score_transaction — prior chargebacks
# ---------------------------------------------------------------------------

def test_multiple_prior_chargebacks_add_20():
    assert score_transaction(_base_tx(prior_chargebacks=2)) == 20


def test_single_prior_chargeback_adds_5():
    assert score_transaction(_base_tx(prior_chargebacks=1)) == 5


def test_no_prior_chargebacks_adds_nothing():
    assert score_transaction(_base_tx(prior_chargebacks=0)) == 0


# ---------------------------------------------------------------------------
# score_transaction — clamping and combined signals
# ---------------------------------------------------------------------------

def test_score_capped_at_100():
    tx = _base_tx(
        device_risk_score=85,
        is_international=1,
        amount_usd=1500,
        velocity_24h=8,
        failed_logins_24h=6,
        prior_chargebacks=3,
    )
    assert score_transaction(tx) == 100


def test_score_floor_at_zero():
    assert score_transaction(_base_tx()) == 0


def test_high_risk_profile_labelled_high():
    tx = _base_tx(device_risk_score=80, is_international=1, velocity_24h=7)
    assert label_risk(score_transaction(tx)) == "high"
