from agents.medical_image_classifier_agent import MedicalImageClassifierAgent

image_path = "dataset/chest_xray/test/PNEUMONIA/person1_virus_6.jpeg"

agent = MedicalImageClassifierAgent()

result = agent.classify(image_path)

print(result)