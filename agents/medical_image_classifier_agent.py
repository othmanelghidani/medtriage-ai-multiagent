from tools.classify_xray_tool import classify_xray_tool
from tools.logger import log_event


class MedicalImageClassifierAgent:
    def __init__(self):
        self.name = "Medical Image Classifier Agent"

    def classify(self, image_path):
        """
        Uses the trained CNN model through classify_xray_tool.
        Returns prediction and confidence.
        """

        print(f"[{self.name}] Starting chest X-ray classification...")

        result = classify_xray_tool(image_path)

        if result.get("error"):
            print(f"[{self.name}] Error: {result['error']}")

            log_event(
                agent=self.name,
                action="Classification failed",
                error=result["error"]
            )

            return {
                "error": result["error"],
                "prediction": None,
                "confidence": 0.0
            }

        print(f"[{self.name}] Classification completed.")
        print(f"[{self.name}] Prediction: {result['prediction']}")
        print(f"[{self.name}] Confidence: {result['confidence']}")

        log_event(
            agent=self.name,
            action="Classified chest X-ray image",
            result={
                "prediction": result["prediction"],
                "confidence": result["confidence"]
            }
        )

        return {
            "error": None,
            "prediction": result["prediction"],
            "confidence": result["confidence"]
        }