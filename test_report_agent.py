from agents.medical_report_generator_agent import MedicalReportGeneratorAgent

agent = MedicalReportGeneratorAgent()

result = agent.generate_report(
    prediction="PNEUMONIA",
    confidence=0.91
)

print(result)