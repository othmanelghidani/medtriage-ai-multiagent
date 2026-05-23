from datetime import datetime


def human_validation_tool(report, decision, reviewer_name="Human Reviewer"):
    """
    Human-in-the-loop validation.
    The final report is only approved if the human reviewer accepts it.
    """

    try:
        if report is None:
            return {
                "error": "Report is missing.",
                "final_report": None
            }

        decision = decision.lower().strip()

        if decision not in ["approve", "reject"]:
            return {
                "error": "Invalid decision. Use 'approve' or 'reject'.",
                "final_report": None
            }

        final_report = report.copy()

        final_report["human_validation_status"] = (
            "Approved" if decision == "approve" else "Rejected"
        )

        final_report["reviewer_name"] = reviewer_name
        final_report["review_timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        if decision == "approve":
            final_report["final_status"] = "Final report approved by human reviewer."
        else:
            final_report["final_status"] = "Report rejected. Medical review required."

        return {
            "error": None,
            "final_report": final_report
        }

    except Exception as e:
        return {
            "error": str(e),
            "final_report": None
        }