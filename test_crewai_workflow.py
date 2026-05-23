from agents.crewai_workflow import MedicalTriageCrew

image_path = "dataset/chest_xray/test/PNEUMONIA/person1_virus_6.jpeg"

crew = MedicalTriageCrew()

result = crew.run(image_path)

print(result)