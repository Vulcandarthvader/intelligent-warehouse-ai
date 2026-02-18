import sys
import os

# Add project root to Python path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJECT_ROOT)

from vision.detect import detect_objects
from ml_model.inference import classify_image
from rag.rag_system import WarehouseRAG


print("Running Vision Module...")
detect_objects(os.path.join(PROJECT_ROOT, "vision/box_clean.jpg"))


print("\nRunning Classification Module...")
test_image = os.path.join(PROJECT_ROOT, "ml_model/data/fragile/box1.jpg")
category = classify_image(test_image)
print("Predicted Category:", category)


print("\nRunning RAG Module...")
rag = WarehouseRAG()

query = f"How should {category} items be handled?"
results = rag.query(query)

print("\nFinal Response:")
for r in results:
    print("\n---", r["document"], "---")
    print(r["content"])

