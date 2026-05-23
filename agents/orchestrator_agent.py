import os

from agents.medical_image_classifier_agent import MedicalImageClassifierAgent
from agents.medical_report_generator_agent import MedicalReportGeneratorAgent
from tools.logger import log_event
from tools.xray_validator import validate_xray_image


class OrchestratorAgent:
    def __init__(self):
        self.name = "Orchestrator Agent"
        self.classifier_agent = MedicalImageClassifierAgent()
        self.report_agent = MedicalReportGeneratorAgent()

    def validate_image_path(self, image_path):
        if not image_path:
            return False, "Image path is missing."

        if not os.path.exists(image_path):
            return False, "Image file does not exist."

        allowed_extensions = [".jpg", ".jpeg", ".png"]
        extension = os.path.splitext(image_path)[1].lower()

        if extension not in allowed_extensions:
            return False, "Invalid image format. Only JPG, JPEG and PNG are supported."

        if os.path.getsize(image_path) == 0:
            return False, "Image file is empty."

        return True, None

    def validate_model_exists(self):
        model_path = "models/model.pth"

        if not os.path.exists(model_path):
            return False, "Trained model not found. Please train the CNN first."

        return True, None

    def check_low_confidence(self, classification_result):
        confidence = classification_result.get("confidence", 0)
        return confidence < 0.60

    def run(self, image_path):
        print(f"[{self.name}] Starting robust workflow...")

        log_event(
            agent=self.name,
            action="Started workflow",
            result={"image_path": image_path}
        )

        is_valid_image, image_error = self.validate_image_path(image_path)

        if not is_valid_image:
            log_event(
                agent=self.name,
                action="Image validation failed",
                error=image_error
            )

            return {
                "error": image_error,
                "classification": None,
                "report": None
            }

        is_xray, xray_error = validate_xray_image(image_path)

        if not is_xray:
            log_event(
                agent=self.name,
                action="X-ray validation failed",
                error=xray_error
            )

            return {
                "error": xray_error,
                "classification": None,
                "report": None
            }

        model_exists, model_error = self.validate_model_exists()

        if not model_exists:
            log_event(
                agent=self.name,
                action="Model validation failed",
                error=model_error
            )

            return {
                "error": model_error,
                "classification": None,
                "report": None
            }

        print(f"[{self.name}] Image, X-ray format and model validated successfully.")

        classification_result = self.classifier_agent.classify(image_path)

        if classification_result.get("error"):
            log_event(
                agent=self.name,
                action="Classification step failed",
                error=classification_result["error"]
            )

            return {
                "error": classification_result["error"],
                "classification": classification_result,
                "report": None
            }

        if self.check_low_confidence(classification_result):
            log_event(
                agent=self.name,
                action="Low confidence detected",
                result={
                    "prediction": classification_result["prediction"],
                    "confidence": classification_result["confidence"]
                }
            )

        report_result = self.report_agent.generate_report(
            prediction=classification_result["prediction"],
            confidence=classification_result["confidence"]
        )

        if report_result.get("error"):
            log_event(
                agent=self.name,
                action="Report generation step failed",
                error=report_result["error"]
            )

            return {
                "error": report_result["error"],
                "classification": classification_result,
                "report": None
            }

        log_event(
            agent=self.name,
            action="Workflow completed successfully",
            result={
                "prediction": classification_result["prediction"],
                "confidence": classification_result["confidence"],
                "risk_level": report_result["report"].get("risk_level")
            }
        )

        print(f"[{self.name}] Workflow completed successfully.")
        print(f"[{self.name}] Report is waiting for human validation.")

        return {
            "error": None,
            "classification": classification_result,
            "report": report_result["report"]
        }