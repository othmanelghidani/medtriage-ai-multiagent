from crewai import Agent, Task, Crew, Process

from agents.medical_image_classifier_agent import MedicalImageClassifierAgent
from agents.medical_report_generator_agent import MedicalReportGeneratorAgent


class MedicalTriageCrew:
    def __init__(self):
        self.classifier_agent_logic = MedicalImageClassifierAgent()
        self.report_agent_logic = MedicalReportGeneratorAgent()

    def run(self, image_path):
        """
        CrewAI workflow:
        Orchestrator
        → Medical Image Classifier Agent
        → Medical Report Generator Agent
        """

        print("[CrewAI Orchestrator] Starting multi-agent workflow...")

        # Step 1: Classifier Agent
        classification_result = self.classifier_agent_logic.classify(image_path)

        if classification_result.get("error"):
            return {
                "error": classification_result["error"],
                "classification": None,
                "report": None
            }

        prediction = classification_result["prediction"]
        confidence = classification_result["confidence"]

        # Step 2: Report Generator Agent
        report_result = self.report_agent_logic.generate_report(
            prediction=prediction,
            confidence=confidence
        )

        if report_result.get("error"):
            return {
                "error": report_result["error"],
                "classification": classification_result,
                "report": None
            }

        print("[CrewAI Orchestrator] Workflow completed successfully.")

        return {
            "error": None,
            "classification": classification_result,
            "report": report_result["report"]
        }


# Optional CrewAI declaration for documentation / defense
classifier_agent = Agent(
    role="Medical Image Classifier Agent",
    goal="Classify chest X-ray images as NORMAL or PNEUMONIA using a trained CNN model.",
    backstory=(
        "This agent specializes in medical image analysis. "
        "It uses a PyTorch CNN model wrapped as a functional tool."
    ),
    verbose=True
)

report_generator_agent = Agent(
    role="Medical Report Generator Agent",
    goal="Interpret the CNN prediction, estimate risk level, and generate a structured preliminary medical report.",
    backstory=(
        "This agent specializes in medical reporting. "
        "It uses an Ollama LLM to generate safe, structured, human-reviewed reports."
    ),
    verbose=True
)

classification_task = Task(
    description="Analyze the uploaded chest X-ray image and return prediction plus confidence.",
    expected_output="A dictionary containing prediction and confidence.",
    agent=classifier_agent
)

report_task = Task(
    description="Use the classification result to estimate risk and generate a structured medical report.",
    expected_output="A structured medical report with risk level, recommendation, disclaimer, and validation status.",
    agent=report_generator_agent
)

medical_triage_crew = Crew(
    agents=[classifier_agent, report_generator_agent],
    tasks=[classification_task, report_task],
    process=Process.sequential,
    verbose=True
)