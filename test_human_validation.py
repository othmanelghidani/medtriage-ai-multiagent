from tools.human_validation_tool import human_validation_tool

sample_report = {
    "analysis_type": "AI-assisted preliminary chest X-ray analysis",
    "prediction": "PNEUMONIA",
    "confidence": 0.91,
    "risk_level": "High Risk",
    "recommendation": "Seek immediate medical attention.",
    "human_validation_status": "Pending",
    "medical_disclaimer": "This is not a final diagnosis."
}

result = human_validation_tool(
    report=sample_report,
    decision="approve",
    reviewer_name="Dr. Reviewer"
)

print(result)