from agents.orchestrator_agent import OrchestratorAgent

image_path = "dataset/chest_xray/test/PNEUMONIA/person1_virus_6.jpeg"

orchestrator = OrchestratorAgent()

result = orchestrator.run(image_path)

print(result)