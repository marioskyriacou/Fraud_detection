import logging
# Initialize logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Example mapping of system identifiers to severity weights
system_id_to_risk = {
    "high_risk_rules": 1.0,
    "medium_risk_rules": 0.7,
    "low_medium_risk_rules": 0.4,
    "low_risk_rules": 0.1
}

# Default fallback weights
DEFAULT_W1 = 0.33
DEFAULT_W2 = 0.33
DEFAULT_W3 = 0.33

# Default fallback thresholds
DEFAULT_LOW = 0.3
DEFAULT_MEDIUM = [0.3, 0.7]
DEFAULT_HIGH = 0.7

MAX_OVERRUN = 2.0

# -------------------------------------------------------------
# Extract dynamic TRM weights + thresholds from JSON
# -------------------------------------------------------------
def extract_trm_weights(trm_data):
    weights_json = trm_data.get("trm_weights", [])

    if not weights_json:
        logger.warning("No TRM weight JSON received → using defaults")
        return DEFAULT_W1, DEFAULT_W2, DEFAULT_W3, DEFAULT_LOW, DEFAULT_MEDIUM, DEFAULT_HIGH

    w = weights_json[0]

    W1 = float(w.get("w1", DEFAULT_W1))
    W2 = float(w.get("w2", DEFAULT_W2))
    W3 = float(w.get("w3", DEFAULT_W3))

    LOW = float(w.get("Low", DEFAULT_LOW))
    MEDIUM = w.get("Medium", DEFAULT_MEDIUM)
    if isinstance(MEDIUM, list) and len(MEDIUM) == 2:
        MED_LOW, MED_HIGH = MEDIUM
    else:
        MED_LOW, MED_HIGH = DEFAULT_MEDIUM

    HIGH = float(w.get("High", DEFAULT_HIGH))

    logger.info(f"Using TRM Weights: W1={W1}, W2={W2}, W3={W3}")
    logger.info(f"Decision thresholds → LOW<{LOW}, MEDIUM={MED_LOW}-{MED_HIGH}, HIGH>{HIGH}")

    return W1, W2, W3, LOW, (MED_LOW, MED_HIGH), HIGH


def compute_trm_scores(trm_data):
    # Retrieve dynamic weight config
    W1, W2, W3, LOW, (MED_LOW, MED_HIGH), HIGH = extract_trm_weights(trm_data)
    logger.info(f"Processing Transaction: {trm_data.get('transaction_reference')}")

    total_rules = trm_data.get("totalActiveRules", 0)
    try:
        total_rules = float(total_rules)
    except Exception:
        total_rules = 0
        logger.warning("totalActiveRules could not be converted to numeric → set to 0")

    fired_rules = trm_data.get("fired_rules", [])
    fired_count = len(fired_rules)

    # Rule Activation Score (S1)
    s1 = fired_count / total_rules if total_rules > 0 else 0
    logger.info(f"Rule Activation Score (S1): Fired {fired_count}/{total_rules} → {s1:.4f}")


    # Severity Score (S2)
    weighted_sum = 0
    for r in fired_rules:
        risk_tag = r.get("risk_value_map", None)
        if risk_tag in system_id_to_risk:
            severity = float(system_id_to_risk[risk_tag])
            logger.info(f"Severity Lookup: {risk_tag} → {severity}")
        else:
            severity = 0
            logger.warning(f"Risk tag not found: {risk_tag} → Severity assigned = 0")
        weighted_sum += severity

    max_severity_weight = 1.0
    s2 = weighted_sum / (fired_count * max_severity_weight) if fired_count > 0 else 0
    logger.info(f"Severity Score (S2): {s2:.4f}")

    # Overrun Score (S3)
    overrun_values = []

    for r in fired_rules:
        numeric_str = r.get("alert_computer_numeric", "")
        if not numeric_str:
            overrun_values.append(0)
            continue

        pairs = numeric_str.split("|")
        pair_scores = []

        for pair in pairs:
            try:
                numbers = [
                    float(s) for s in pair.replace("=", " ").replace(">", " ").replace("<", " ").split()
                    if s.replace('.', '', 1).isdigit()
                ]

                if len(numbers) >= 2:
                    actual, expected = numbers[0], numbers[1]
                    if expected != 0:
                        raw_ratio = actual / expected
                        deviation = abs(raw_ratio - 1)
                        normalized = deviation / MAX_OVERRUN
                        normalized = max(0, min(normalized, 1))
                    else:
                        normalized = 0
                    pair_scores.append(normalized)

                elif len(numbers) == 1:
                    # Only one numeric value → normalized = 1  for the binary categories
                    pair_scores.append(1)

                else:
                    logger.warning(f"Skipping pair due to insufficient numeric values: '{pair}'")

            except Exception as e:
                logger.warning(f"Skipping malformed pair '{pair}' → {e}")

        # Average normalized score for this rule
        rule_overrun = sum(pair_scores) / len(pair_scores) if pair_scores else 0
        overrun_values.append(rule_overrun)

    # Average over all rules to get S3
    s3 = sum(overrun_values) / len(overrun_values) if overrun_values else 0
    logger.info(f"Overrun Score (S3 normalized): {s3:.4f}")

    # Weighted TRM score
    trm_score = W1 * s1 + W2 * s2 + W3 * s3
    logger.info(f"TRM Weighted Score: {trm_score:.4f}")

    return {
        "transaction_reference": trm_data.get("transaction_reference"),
        "decision_numeric": None,
        "decision_label": trm_data.get("rule_engine_decision"),
        "thresholds": {"low": LOW, "medium": (MED_LOW, MED_HIGH), "high": HIGH},
        "scores": {
            "rule_activation_score": round(s1, 4),
            "severity_score": round(s2, 4),
            "overrun_score": round(s3, 4),
            "trm_risk_score": round(trm_score, 4),
            "final_combined_score": round(trm_score, 4)
        }
    }

def make_final_decision(trm_data, ml_data):
    trm_result = compute_trm_scores(trm_data)

    s1 = trm_result["scores"]["rule_activation_score"]
    s2 = trm_result["scores"]["severity_score"]
    s3 = trm_result["scores"]["overrun_score"]
    trm_score = trm_result["scores"]["trm_risk_score"]
    LOW = trm_result["thresholds"]["low"]
    MED_LOW, MED_HIGH = trm_result["thresholds"]["medium"]
    HIGH = trm_result["thresholds"]["high"]
    ml_score = ml_data.get("Model_Confidence", 0)
    ml_prediction = ml_data.get("Model_Prediction")

    logger.info(f"TRM Final Scores → S1={s1}, S2={s2}, S3={s3}, TRM Score={trm_score}")

    # --- TRM decision mapping ---
    if trm_score < LOW:
        trm_decision_label = "COMPLIANCE"
    elif trm_score <  HIGH:
        trm_decision_label = "REVIEW"
    else:
        trm_decision_label = "NON-COMPLIANCE"

    ml_label = "NON-COMPLIANCE" if ml_prediction == 1 else "COMPLIANCE"

    if ml_label == trm_decision_label:
        if ml_score >= 0.7:
            decision_label = ml_label
            explanation = "ML,TRM agree ,ML confidence high -- unified decision"
        else:
            decision_label = trm_decision_label
            explanation = "ML, TRM agree but ML confidence low -- keep TRM decision"
    else:
        decision_label = "Manual Review"
        explanation = "ML and TRM disagree --- manual review required"
    decision_cfg = trm_data.get("decision_weights", [])

    if decision_cfg:  # Only apply if not empty
        cfg = decision_cfg[0]

        # Use ONLY the client-provided values
        trm_w = float(cfg["trm_weight"])
        ml_w = float(cfg["ml_weight"])
        threshold = float(cfg["given_threshold"])

        # Compute final score
        final_score = (trm_w * trm_score) + (ml_w * ml_score)

        # Compliance decision
        decision_label = "COMPLIANCE" if final_score < threshold else "NON-COMPLIANCE"

        explanation = (
            "Decision based on weighted TRM+ML final score compared with client threshold"
        )
    return {
        "transaction_reference": trm_data.get("transaction_reference"),
        "Final_Decision": decision_label,
        "Explanation": explanation,
        "Reason": {
            "ml_confidence": round(ml_score, 4),
            "ml_prediction": ml_prediction,
            "ml_explainability": ml_data.get("Transaction_Details_categorical", {}),
            "Rule_engine_risk_score": round(trm_score, 4),
        },
    }
