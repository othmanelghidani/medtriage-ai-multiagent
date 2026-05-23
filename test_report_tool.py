from tools.generate_report_tool import generate_medical_report_tool

result = generate_medical_report_tool(
    prediction="PNEUMONIA",
    confidence=0.91
)

print(result)