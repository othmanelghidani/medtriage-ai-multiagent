from tools.classify_xray_tool import classify_xray_tool

image_path = "dataset/chest_xray/test/NORMAL/IM-0001-0001.jpeg"

result = classify_xray_tool(image_path)

print(result)