ESCALATION_RESPONSE = (
    "Thank you for reaching out. Your request requires attention from a human support agent "
    "who will follow up with you shortly. We apologize for any inconvenience."
)


def decide(
    rule_triggered: bool,
    company_unknown: bool,
    no_corpus_match: bool,
    risk_level: str,
    generator_status: str,
    grounded: bool,
) -> tuple[str, str]:
    """
    Aggregate all pipeline signals and return (status, escalation_reason).
    Priority: rule > company_unknown > no_corpus_match > high_risk > not_grounded > generator_status.
    Returns ("escalated"|"replied", reason_string).
    """
    if rule_triggered:
        return "escalated", "High-risk keyword detected in ticket. Routed to human agent."

    if company_unknown:
        return "escalated", "Company could not be determined from ticket content. Escalated to avoid wrong-domain response."

    if no_corpus_match:
        return "escalated", "No relevant support article found in corpus. Cannot ground a response."

    if generator_status == "escalated":
        return "escalated", "LLM determined the ticket requires human handling."

    return "replied", ""


def escalation_response() -> str:
    return ESCALATION_RESPONSE
