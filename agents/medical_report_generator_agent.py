from tools.generate_report_tool import generate_medical_report_tool
from tools.logger import log_event


class MedicalReportGeneratorAgent:
    def __init__(self):
        self.name = "Medical Report Generator Agent"

    def generate_report(self, prediction, confidence):
        """
        Uses the LLM report generation tool.
        It interprets the CNN result, estimates risk, and generates a structured report.
        """

        print(f"[{self.name}] Starting medical report generation...")

        if prediction is None:
            error_message = "Prediction is missing."

            log_event(
                agent=self.name,
                action="Report generation failed",
                error=error_message
            )

            return {
                "error": error_message,
                "report": None
            }

        result = generate_medical_report_tool(
            prediction=prediction,
            confidence=confidence
        )

        if result.get("error"):
            print(f"[{self.name}] Error: {result['error']}")

            log_event(
                agent=self.name,
                action="Report generation failed",
                error=result["error"]
            )

            return {
                "error": result["error"],
                "report": None
            }

        print(f"[{self.name}] Report generated successfully.")

        log_event(
            agent=self.name,
            action="Generated medical report",
            result={
                "prediction": result["report"].get("prediction"),
                "confidence": result["report"].get("confidence"),
                "risk_level": result["report"].get("risk_level"),
                "generation_mode": result["report"].get("generation_mode")
            }
        )

        return {
            "error": None,
            "report": result["report"]
        }