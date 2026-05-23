import json
from datetime import datetime
import ollama


def generate_medical_report_tool(prediction, confidence):

    try:

        prompt = f"""
You are a medical triage assistant inside a multi-agent AI system.

The CNN image classification agent returned:
- Prediction: {prediction}
- Confidence: {confidence}

Your tasks:
1. Estimate the medical triage risk level.
2. Generate a short medical preliminary report.
3. Provide a safe recommendation.
4. Mention that this is NOT a final diagnosis.
5. Mention that human validation is required.

Risk rules:
- Pneumonia + confidence >= 0.85 => High Risk
- Pneumonia + confidence >= 0.60 => Medium Risk
- Normal + confidence >= 0.80 => Low Risk
- Otherwise => Uncertain Risk

Return ONLY valid JSON in this exact format:

{{
    "analysis_type": "AI-assisted preliminary chest X-ray analysis",
    "prediction": "{prediction}",
    "confidence": {confidence},
    "risk_level": "",
    "recommendation": "",
    "human_validation_status": "Pending",
    "medical_disclaimer": ""
}}
"""

        response = ollama.chat(
            model='phi3',
            messages=[
                {
                    'role': 'user',
                    'content': prompt
                }
            ]
        )

        raw_text = response['message']['content'].strip()

        # Remove markdown formatting if present
        raw_text = raw_text.replace("```json", "")
        raw_text = raw_text.replace("```", "")
        raw_text = raw_text.strip()

        report = json.loads(raw_text)

        report["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        report["generation_mode"] = "Ollama"

        return {
            "error": None,
            "report": report
        }

    except Exception as e:
        return {
            "error": str(e),
            "report": None
        }