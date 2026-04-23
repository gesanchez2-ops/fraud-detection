import pandas as pd
import pytest

from features import build_model_frame
from analyze_fraud import load_inputs, score_transactions, summarize_results


# ---------------------------------------------------------------------------
# build_model_frame
# ---------------------------------------------------------------------------

def _minimal_frames(amount_usd=1200.0, failed_logins_24h=3, prior_chargebacks=1):
    transactions = pd.DataFrame([{
        "transaction_id": 1,
        "account_id": 100,
        "amount_usd": amount_usd,
        "failed_logins_24h": failed_logins_24h,
    }])
    accounts = pd.DataFrame([{
        "account_id": 100,
        "prior_chargebacks": prior_chargebacks,
    }])
    return transactions, accounts


def test_build_model_frame_merges_account_data():
    transactions, accounts = _minimal_frames(prior_chargebacks=2)
    result = build_model_frame(transactions, accounts)
    assert "prior_chargebacks" in result.columns
    assert result.iloc[0]["prior_chargebacks"] == 2


def test_is_large_amount_true_at_1000():
    transactions, accounts = _minimal_frames(amount_usd=1000.0)
    result = build_model_frame(transactions, accounts)
    assert result.iloc[0]["is_large_amount"] == 1


def test_is_large_amount_false_below_1000():
    transactions, accounts = _minimal_frames(amount_usd=999.99)
    result = build_model_frame(transactions, accounts)
    assert result.iloc[0]["is_large_amount"] == 0


def test_login_pressure_none():
    transactions, accounts = _minimal_frames(failed_logins_24h=0)
    result = build_model_frame(transactions, accounts)
    assert str(result.iloc[0]["login_pressure"]) == "none"


def test_login_pressure_low():
    transactions, accounts = _minimal_frames(failed_logins_24h=1)
    result = build_model_frame(transactions, accounts)
    assert str(result.iloc[0]["login_pressure"]) == "low"


def test_login_pressure_high():
    transactions, accounts = _minimal_frames(failed_logins_24h=5)
    result = build_model_frame(transactions, accounts)
    assert str(result.iloc[0]["login_pressure"]) == "high"


# ---------------------------------------------------------------------------
# Full pipeline — real data
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def pipeline():
    accounts, transactions, chargebacks = load_inputs()
    scored = score_transactions(transactions, accounts)
    summary = summarize_results(scored, chargebacks)
    return scored, summary, chargebacks


def test_scored_frame_has_risk_columns(pipeline):
    scored, _, _ = pipeline
    assert "risk_score" in scored.columns
    assert "risk_label" in scored.columns


def test_all_transactions_are_scored(pipeline):
    scored, _, _ = pipeline
    assert scored["risk_score"].notna().all()
    assert scored["risk_label"].notna().all()


def test_summary_sort_order_is_low_medium_high(pipeline):
    _, summary, _ = pipeline
    assert list(summary["risk_label"]) == ["low", "medium", "high"]


def test_high_tier_chargeback_rate_is_1(pipeline):
    _, summary, _ = pipeline
    high_row = summary[summary["risk_label"] == "high"].iloc[0]
    assert high_row["chargeback_rate"] == 1.0


def test_low_tier_has_no_chargebacks(pipeline):
    _, summary, _ = pipeline
    low_row = summary[summary["risk_label"] == "low"].iloc[0]
    assert low_row["chargebacks"] == 0


def test_no_known_fraud_transaction_scores_low(pipeline):
    scored, _, chargebacks = pipeline
    fraud_ids = set(chargebacks["transaction_id"])
    fraud_rows = scored[scored["transaction_id"].isin(fraud_ids)]
    assert (fraud_rows["risk_label"] != "low").all(), (
        "A confirmed chargeback transaction was labelled low risk"
    )


def test_all_chargebacks_appear_in_summary(pipeline):
    scored, summary, chargebacks = pipeline
    matched_chargebacks = chargebacks[chargebacks["transaction_id"].isin(scored["transaction_id"])]
    assert summary["chargebacks"].sum() == len(matched_chargebacks)
