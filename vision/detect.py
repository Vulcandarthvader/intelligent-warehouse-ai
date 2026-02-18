import cv2
import numpy as np


def detect_objects(image_path):
    image = cv2.imread(image_path)

    if image is None:
        print("Could not load image.")
        return []

    output = image.copy()

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(blurred, 50, 150)

    kernel = np.ones((5, 5), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detections = []

    for cnt in contours:
        area = cv2.contourArea(cnt)

        if area > 10000:
            x, y, w, h = cv2.boundingRect(cnt)

            center_x = x + w // 2
            center_y = y + h // 2

            detections.append({
                "bbox": (x, y, w, h),
                "dimensions": (w, h),
                "center": (center_x, center_y)
            })

            cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.circle(output, (center_x, center_y), 5, (0, 0, 255), -1)

            label = f"{w}x{h}"
            cv2.putText(
                output,
                label,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 0, 0),
                2
            )

    # Save result instead of showing GUI
    output_path = "results/detected_output.jpg"
    cv2.imwrite(output_path, output)
    print(f"Saved detection result to {output_path}")

    return detections

