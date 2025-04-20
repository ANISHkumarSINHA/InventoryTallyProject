# import cv2
# import numpy as np
# import imutils
# import requests
# from flask import Flask, render_template, Response
# from ultralytics import YOLO

# app = Flask(__name__, template_folder='../templates', static_folder='../static')

# # Load Models
# count_model = YOLO('C:/Users/ANISH SINHA/OneDrive/Desktop/New folder/InventoryTallyProject/scripts/models/sku3/weights/best.pt')  # SKU (for total count)
# loreal_model = YOLO('C:/Users/ANISH SINHA/OneDrive/Desktop/New folder/InventoryTallyProject/scripts/models/loreal/weights/best.pt')  # Loreal
# egg_model = YOLO('C:/Users/ANISH SINHA/OneDrive/Desktop/New folder/InventoryTallyProject/scripts/models/egg_detection3/weights/best.pt')  # Egg Detection
# drinks_model = YOLO('C:/Users/ANISH SINHA/OneDrive/Desktop/New folder/InventoryTallyProject/scripts/models/drinks2/weights/best.pt')  # Drinks Detection

# url = "http://192.168.134.126:8080/shot.jpg"

# # Prediction Functions
# def predict_inventory(frame):
#     return count_model.predict(frame, conf=0.25)

# def predict_loreal(frame):
#     return loreal_model.predict(frame, conf=0.5)

# def predict_egg(frame):
#     return egg_model.predict(frame, conf=0.5)

# def predict_drinks(frame):
#     return drinks_model.predict(frame, conf=0.5)

# def count(results):
#     return len(results[0].boxes) if results else 0

# # Streaming with Detection
# def camera_stream():
#     while True:
#         img_resp = requests.get(url)
#         img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8)
#         img = cv2.imdecode(img_arr, -1)
#         img = imutils.resize(img, width=1000)

#         # Predictions
#         result_count = predict_inventory(img)  # SKU for total count
#         result_loreal = predict_loreal(img)
#         result_egg = predict_egg(img)
#         result_drinks = predict_drinks(img)

#         # Total count from SKU model only
#         total_count = count(result_count)

#         # Draw bounding boxes with custom labels and colors
#         # SKU (Green, no specific label unless provided)
#         for r in result_count:
#             for box in r.boxes:
#                 x1, y1, x2, y2 = map(int, box.xyxy[0])
#                 label = r.names[int(box.cls)] if r.names else "SKU"  # Generic label if no specific class
#                 cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green
#                 cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

#         # Loreal (Blue, retain detailed labels)
#         for r in result_loreal:
#             for box in r.boxes:
#                 x1, y1, x2, y2 = map(int, box.xyxy[0])
#                 label = r.names[int(box.cls)]  # Retains detailed labels like "Loreal Shampoo 100ml"
#                 cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Blue
#                 cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

#         # Eggs (Blue, label as "Egg")
#         for r in result_egg:
#             for box in r.boxes:
#                 x1, y1, x2, y2 = map(int, box.xyxy[0])
#                 label = "Egg"  # Fixed label for all eggs
#                 cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Blue
#                 cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

#         # Drinks (Orange, label as "Colddrink" assuming it's a class)
#         for r in result_drinks:
#             for box in r.boxes:
#                 x1, y1, x2, y2 = map(int, box.xyxy[0])
#                 label = "Colddrink" if "colddrink" in r.names[int(box.cls)].lower() else r.names[int(box.cls)]  # Adjust based on dataset
#                 cv2.rectangle(img, (x1, y1), (x2, y2), (255, 165, 0), 2)  # Orange
#                 cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 165, 0), 2)

#         # Text overlay (only total count from SKU on the left side)
#         cv2.putText(img, f"{total_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)  # Total count in green

#         _, buffer = cv2.imencode('.jpg', img)
#         frame = buffer.tobytes()
#         yield (b'--frame\r\n'
#                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/video_feed')
# def video_feed():
#     return Response(camera_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')

# if __name__ == "__main__":
#     app.run(debug=True, port=5000)

#⭐⭐⭐⭐
# import cv2
# import numpy as np
# import imutils
# import requests
# from flask import Flask, render_template, Response
# from ultralytics import YOLO

# app = Flask(__name__, template_folder='../templates', static_folder='../static')

# # Load Models
# count_model = YOLO('C:/Users/ANISH SINHA/OneDrive/Desktop/New folder/InventoryTallyProject/scripts/models/sku3/weights/best.pt')  # SKU (for total count)
# loreal_model = YOLO('C:/Users/ANISH SINHA/OneDrive/Desktop/New folder/InventoryTallyProject/scripts/models/loreal/weights/best.pt')  # Loreal
# egg_model = YOLO('C:/Users/ANISH SINHA/OneDrive/Desktop/New folder/InventoryTallyProject/scripts/models/egg_detection3/weights/best.pt')  # Egg Detection
# drinks_model = YOLO('C:/Users/ANISH SINHA/OneDrive/Desktop/New folder/InventoryTallyProject/scripts/models/drinks2/weights/best.pt')  # Drinks Detection
# cold_drinks_model = YOLO('C:/Users/ANISH SINHA/OneDrive/Desktop/New folder/InventoryTallyProject/scripts/models/cold_drinks_detection4/weights/best.pt')  # New Cold Drinks Detection

# url = "http://192.168.134.126:8080/shot.jpg"

# # Prediction Functions
# def predict_inventory(frame):
#     return count_model.predict(frame, conf=0.25)

# def predict_loreal(frame):
#     return loreal_model.predict(frame, conf=0.5)

# def predict_egg(frame):
#     return egg_model.predict(frame, conf=0.5)

# def predict_drinks(frame):
#     return drinks_model.predict(frame, conf=0.5)

# def predict_cold_drinks(frame):
#     return cold_drinks_model.predict(frame, conf=0.5)

# def count(results):
#     return len(results[0].boxes) if results else 0

# # Streaming with Detection
# def camera_stream():
#     while True:
#         img_resp = requests.get(url)
#         img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8)
#         img = cv2.imdecode(img_arr, -1)
#         img = imutils.resize(img, width=1000)

#         # Predictions
#         result_count = predict_inventory(img)  # SKU for total count
#         result_loreal = predict_loreal(img)
#         result_egg = predict_egg(img)
#         result_drinks = predict_drinks(img)
#         result_cold_drinks = predict_cold_drinks(img)

#         # Total count from SKU model only
#         total_count = count(result_count)

#         # Draw bounding boxes with custom labels and colors
#         # SKU (Green, no specific label unless provided)
#         for r in result_count:
#             for box in r.boxes:
#                 x1, y1, x2, y2 = map(int, box.xyxy[0])
#                 label = r.names[int(box.cls)] if r.names else "SKU"  # Generic label if no specific class
#                 cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green
#                 cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

#         # Loreal (Blue, retain detailed labels)
#         for r in result_loreal:
#             for box in r.boxes:
#                 x1, y1, x2, y2 = map(int, box.xyxy[0])
#                 label = r.names[int(box.cls)]  # Retains detailed labels like "Loreal Shampoo 100ml"
#                 cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Blue
#                 cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

#         # Eggs (Blue, label as "Egg")
#         for r in result_egg:
#             for box in r.boxes:
#                 x1, y1, x2, y2 = map(int, box.xyxy[0])
#                 label = "Egg"  # Fixed label for all eggs
#                 cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Blue
#                 cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

#         # Drinks (Orange, label as "Colddrink" assuming it's a class)
#         for r in result_drinks:
#             for box in r.boxes:
#                 x1, y1, x2, y2 = map(int, box.xyxy[0])
#                 label = "Colddrink" if "colddrink" in r.names[int(box.cls)].lower() else r.names[int(box.cls)]  # Adjust based on dataset
#                 cv2.rectangle(img, (x1, y1), (x2, y2), (255, 165, 0), 2)  # Orange
#                 cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 165, 0), 2)

#         # New Cold Drinks (Purple, label based on dataset classes)
#         for r in result_cold_drinks:
#             for box in r.boxes:
#                 x1, y1, x2, y2 = map(int, box.xyxy[0])
#                 cls_id = int(box.cls)
#                 if 0 <= cls_id < 6:  # Ensure class ID is within range (0 to 5 for 6 classes)
#                     label = ['Coca Cola', 'Sprite', 'Pepsi', 'Mountain Dew', '7UP', 'Fanta'][cls_id]
#                 else:
#                     label = "Colddrink"  # Fallback if class ID is invalid
#                 cv2.rectangle(img, (x1, y1), (x2, y2), (128, 0, 128), 2)  # Purple
#                 cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 0, 128), 2)

#         # Text overlay (only total count from SKU on the left side)
#         cv2.putText(img, f"{total_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)  # Total count in green

#         _, buffer = cv2.imencode('.jpg', img)
#         frame = buffer.tobytes()
#         yield (b'--frame\r\n'
#                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/video_feed')
# def video_feed():
#     return Response(camera_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')

# if __name__ == "__main__":
#     app.run(debug=True, port=5000)

# import cv2
# import numpy as np
# import imutils
# import requests
# from flask import Flask, render_template, Response
# from ultralytics import YOLO

# app = Flask(__name__, template_folder='../templates', static_folder='../static')

# # Load Models
# count_model = YOLO('C:/Users/ANISH SINHA/OneDrive/Desktop/New folder/InventoryTallyProject/scripts/models/sku3/weights/best.pt')  # SKU (for total count)
# loreal_model = YOLO('C:/Users/ANISH SINHA/OneDrive/Desktop/New folder/InventoryTallyProject/scripts/models/loreal/weights/best.pt')  # Loreal
# egg_model = YOLO('C:/Users/ANISH SINHA/OneDrive/Desktop/New folder/InventoryTallyProject/scripts/models/egg_detection3/weights/best.pt')  # Egg Detection
# drinks_model = YOLO('C:/Users/ANISH SINHA/OneDrive/Desktop/New folder/InventoryTallyProject/scripts/models/drinks2/weights/best.pt')  # Drinks Detection
# cold_drinks_model = YOLO('C:/Users/ANISH SINHA/OneDrive/Desktop/New folder/InventoryTallyProject/scripts/models/cold_drinks_detection4/weights/best.pt')  # New Cold Drinks Detection
# toothpaste_model = YOLO('C:/Users/ANISH SINHA/OneDrive/Desktop/New folder/InventoryTallyProject/scripts/models/toothpaste_detection/weights/best.pt')  # Toothpaste Detection

# url = "http://192.168.134.126:8080/shot.jpg"

# # Prediction Functions
# def predict_inventory(frame):
#     return count_model.predict(frame, conf=0.25)

# def predict_loreal(frame):
#     return loreal_model.predict(frame, conf=0.5)

# def predict_egg(frame):
#     return egg_model.predict(frame, conf=0.5)

# def predict_drinks(frame):
#     return drinks_model.predict(frame, conf=0.5)

# def predict_cold_drinks(frame):
#     return cold_drinks_model.predict(frame, conf=0.5)

# def predict_toothpaste(frame):
#     return toothpaste_model.predict(frame, conf=0.5)

# def count(results):
#     return len(results[0].boxes) if results else 0

# # Streaming with Detection
# def camera_stream():
#     while True:
#         img_resp = requests.get(url)
#         img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8)
#         img = cv2.imdecode(img_arr, -1)
#         img = imutils.resize(img, width=1000)

#         # Predictions
#         result_count = predict_inventory(img)  # SKU for total count
#         result_loreal = predict_loreal(img)
#         result_egg = predict_egg(img)
#         result_drinks = predict_drinks(img)
#         result_cold_drinks = predict_cold_drinks(img)
#         result_toothpaste = predict_toothpaste(img)

#         # Total count from SKU model only
#         total_count = count(result_count)

#         # Draw bounding boxes with custom labels and colors
#         # SKU (Green, no specific label unless provided)
#         for r in result_count:
#             for box in r.boxes:
#                 x1, y1, x2, y2 = map(int, box.xyxy[0])
#                 label = r.names[int(box.cls)] if r.names else "SKU"
#                 cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
#                 cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

#         # Loreal (Blue, retain detailed labels)
#         for r in result_loreal:
#             for box in r.boxes:
#                 x1, y1, x2, y2 = map(int, box.xyxy[0])
#                 label = r.names[int(box.cls)]
#                 cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
#                 cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

#         # Eggs (Blue, label as "Egg")
#         for r in result_egg:
#             for box in r.boxes:
#                 x1, y1, x2, y2 = map(int, box.xyxy[0])
#                 label = "Egg"
#                 cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
#                 cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

#         # Drinks (Orange, label as "Colddrink" assuming it's a class)
#         for r in result_drinks:
#             for box in r.boxes:
#                 x1, y1, x2, y2 = map(int, box.xyxy[0])
#                 label = "Colddrink" if "colddrink" in r.names[int(box.cls)].lower() else r.names[int(box.cls)]
#                 cv2.rectangle(img, (x1, y1), (x2, y2), (255, 165, 0), 2)
#                 cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 165, 0), 2)

#         # New Cold Drinks (Purple, label based on dataset classes)
#         for r in result_cold_drinks:
#             for box in r.boxes:
#                 x1, y1, x2, y2 = map(int, box.xyxy[0])
#                 cls_id = int(box.cls)
#                 if 0 <= cls_id < 6:
#                     label = ['Coca Cola', 'Sprite', 'Pepsi', 'Mountain Dew', '7UP', 'Fanta'][cls_id]
#                 else:
#                     label = "Colddrink"
#                 cv2.rectangle(img, (x1, y1), (x2, y2), (128, 0, 128), 2)
#                 cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 0, 128), 2)

#         # Toothpaste (Teal, label based on dataset)
#         for r in result_toothpaste:
#             for box in r.boxes:
#                 x1, y1, x2, y2 = map(int, box.xyxy[0])
#                 cls_id = int(box.cls)
#                 if 0 <= cls_id < 1:
#                     label = ['toothpaste'][cls_id]
#                 else:
#                     label = "Item"
#                 cv2.rectangle(img, (x1, y1), (x2, y2), (0, 128, 128), 2)
#                 cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 128, 128), 2)

#         # Text overlay (only total count from SKU on the left side)
#         cv2.putText(img, f"{total_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

#         _, buffer = cv2.imencode('.jpg', img)
#         frame = buffer.tobytes()
#         yield (b'--frame\r\n'
#                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/video_feed')
# def video_feed():
#     return Response(camera_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')

# if __name__ == "__main__":
#     app.run(debug=True, port=5000)

# import cv2
# import numpy as np
# import imutils
# import requests
# from flask import Flask, render_template, Response
# from ultralytics import YOLO

# app = Flask(_name_, template_folder='../templates', static_folder='../static')

# # Load Models
# count_model = YOLO('C:/Users/ANISH SINHA/OneDrive/Desktop/New folder/InventoryTallyProject/scripts/models/sku3/weights/best.pt')  # SKU (for total count)
# loreal_model = YOLO('C:/Users/ANISH SINHA/OneDrive/Desktop/New folder/InventoryTallyProject/scripts/models/loreal/weights/best.pt')  # Loreal
# egg_model = YOLO('C:/Users/ANISH SINHA/OneDrive/Desktop/New folder/InventoryTallyProject/scripts/models/egg_detection3/weights/best.pt')  # Egg Detection
# drinks_model = YOLO('C:/Users/ANISH SINHA/OneDrive/Desktop/New folder/InventoryTallyProject/scripts/models/drinks2/weights/best.pt')  # Drinks Detection
# cold_drinks_model = YOLO('C:/Users/ANISH SINHA/OneDrive/Desktop/New folder/InventoryTallyProject/scripts/models/cold_drinks_detection4/weights/best.pt')  # New Cold Drinks Detection
# toothpaste_model = YOLO('C:/Users/ANISH SINHA/OneDrive/Desktop/New folder/InventoryTallyProject/scripts/models/toothpaste_detection/weights/best.pt')  # Toothpaste Detection
# #grocery_model = YOLO('C:/Users/ANISH SINHA/OneDrive/Desktop/New folder/InventoryTallyProject/scripts/models/grocery_detection/weights/best.pt')  # Grocery Detection

# url = "http://192.168.134.126:8080/shot.jpg"

# # Prediction Functions
# def predict_inventory(frame):
#     return count_model.predict(frame, conf=0.15)

# def predict_loreal(frame):
#     return loreal_model.predict(frame, conf=0.7)

# def predict_egg(frame):
#     return egg_model.predict(frame, conf=0.65)

# def predict_drinks(frame):
#     return drinks_model.predict(frame, conf=0.30)

# def predict_cold_drinks(frame):
#     return cold_drinks_model.predict(frame, conf=0.30)

# def predict_toothpaste(frame):
#     return toothpaste_model.predict(frame, conf=0.65)

# # def predict_grocery(frame):
# #     return grocery_model.predict(frame, conf=0.5)

# def count(results):
#     return len(results[0].boxes) if results else 0

# # Streaming with Detection
# def camera_stream():
#     while True:
#         img_resp = requests.get(url)
#         img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8)
#         img = cv2.imdecode(img_arr, -1)
#         img = imutils.resize(img, width=1000)

#         # Predictions
#         result_count = predict_inventory(img)  # SKU for total count
#         result_loreal = predict_loreal(img)
#         result_egg = predict_egg(img)
#         result_drinks = predict_drinks(img)
#         result_cold_drinks = predict_cold_drinks(img)
#         result_toothpaste = predict_toothpaste(img)
#         #result_grocery = predict_grocery(img)

#         # Total count from SKU model only
#         total_count = count(result_count)

#         # Draw bounding boxes with custom labels and colors
#         # SKU (Green, no specific label unless provided)
#         for r in result_count:
#             for box in r.boxes:
#                 x1, y1, x2, y2 = map(int, box.xyxy[0])
#                 label = r.names[int(box.cls)] if r.names else "SKU"
#                 cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
#                 cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

#         # Loreal (Blue, retain detailed labels)
#         for r in result_loreal:
#             for box in r.boxes:
#                 x1, y1, x2, y2 = map(int, box.xyxy[0])
#                 label = r.names[int(box.cls)]
#                 cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
#                 cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

#         # Eggs (Blue, label as "Egg")
#         for r in result_egg:
#             for box in r.boxes:
#                 x1, y1, x2, y2 = map(int, box.xyxy[0])
#                 label = "Egg"
#                 cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
#                 cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

#         # Drinks (Orange, label as "Colddrink" assuming it's a class)
#         for r in result_drinks:
#             for box in r.boxes:
#                 x1, y1, x2, y2 = map(int, box.xyxy[0])
#                 label = "Colddrink" if "colddrink" in r.names[int(box.cls)].lower() else r.names[int(box.cls)]
#                 cv2.rectangle(img, (x1, y1), (x2, y2), (255, 165, 0), 2)
#                 cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 165, 0), 2)

#         # New Cold Drinks (Purple, label based on dataset classes)
#         for r in result_cold_drinks:
#             for box in r.boxes:
#                 x1, y1, x2, y2 = map(int, box.xyxy[0])
#                 cls_id = int(box.cls)
#                 if 0 <= cls_id < 6:
#                     label = ['Coca Cola', 'Sprite', 'Pepsi', 'Mountain Dew', '7UP', 'Fanta'][cls_id]
#                 else:
#                     label = "Colddrink"
#                 cv2.rectangle(img, (x1, y1), (x2, y2), (128, 0, 128), 2)
#                 cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 0, 128), 2)

#         # Toothpaste (Teal, label based on dataset)
#         for r in result_toothpaste:
#             for box in r.boxes:
#                 x1, y1, x2, y2 = map(int, box.xyxy[0])
#                 cls_id = int(box.cls)
#                 if 0 <= cls_id < 1:
#                     label = ['toothpaste'][cls_id]
#                 else:
#                     label = "Item"
#                 cv2.rectangle(img, (x1, y1), (x2, y2), (0, 128, 128), 2)
#                 cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 128, 128), 2)

#         # # Grocery (Olive, label based on dataset - update with actual classes from data.yaml)
#         # for r in result_grocery:
#         #     for box in r.boxes:
#         #         x1, y1, x2, y2 = map(int, box.xyxy[0])
#         #         cls_id = int(box.cls)
#         #         if 0 <= cls_id < 10:  # Placeholder for 10 classes, adjust based on data.yaml
#         #             label = ['item1', 'item2', 'item3', 'item4', 'item5', 'item6', 'item7', 'item8', 'item9', 'item10'][cls_id]
#         #         else:
#         #             label = "Grocery"
#         #         cv2.rectangle(img, (x1, y1), (x2, y2), (128, 128, 0), 2)  # Olive color
#         #         cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 0), 2)

#         # Text overlay (only total count from SKU on the left side)
#         cv2.putText(img, f"{total_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

#         _, buffer = cv2.imencode('.jpg', img)
#         frame = buffer.tobytes()
#         yield (b'--frame\r\n'
#                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/video_feed')
# def video_feed():
#     return Response(camera_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')

# if _name_ == "_main_":
#     app.run(debug=True, port=5000)

# import cv2
# import numpy as np
# import imutils
# import requests
# from flask import Flask, render_template, Response
# from ultralytics import YOLO

# app = Flask(__name__, template_folder='../templates', static_folder='../static')

# # Load Models
# count_model = YOLO('C:/Users/ANISH SINHA/OneDrive/Desktop/New folder/InventoryTallyProject/scripts/models/sku3/weights/best.pt')  # SKU (for total count)
# loreal_model = YOLO('C:/Users/ANISH SINHA/OneDrive/Desktop/New folder/InventoryTallyProject/scripts/models/loreal/weights/best.pt')  # Loreal
# egg_model = YOLO('C:/Users/ANISH SINHA/OneDrive/Desktop/New folder/InventoryTallyProject/scripts/models/egg_detection3/weights/best.pt')  # Egg Detection
# drinks_model = YOLO('C:/Users/ANISH SINHA/OneDrive/Desktop/New folder/InventoryTallyProject/scripts/models/drinks2/weights/best.pt')  # Drinks Detection
# cold_drinks_model = YOLO('C:/Users/ANISH SINHA/OneDrive/Desktop/New folder/InventoryTallyProject/scripts/models/cold_drinks_detection4/weights/best.pt')  # New Cold Drinks Detection
# toothpaste_model = YOLO('C:/Users/ANISH SINHA/OneDrive/Desktop/New folder/InventoryTallyProject/scripts/models/toothpaste_detection/weights/best.pt')  # Toothpaste Detection
# grocery_model = YOLO('C:/Users/ANISH SINHA/OneDrive/Desktop/New folder/InventoryTallyProject/scripts/models/grocery_detection2/weights/best.pt')  # Grocery Detection

# url = "http://192.168.34.33:8080/shot.jpg"

# # Prediction Functions
# def predict_inventory(frame):
#     return count_model.predict(frame, conf=0.15)

# def predict_loreal(frame):
#     return loreal_model.predict(frame, conf=0.7)

# def predict_egg(frame):
#     return egg_model.predict(frame, conf=0.65)

# def predict_drinks(frame):
#     return drinks_model.predict(frame, conf=0.30)

# def predict_cold_drinks(frame):
#     return cold_drinks_model.predict(frame, conf=0.30)

# def predict_toothpaste(frame):
#     return toothpaste_model.predict(frame, conf=0.65)

# def predict_grocery(frame):
#     return grocery_model.predict(frame, conf=0.5)

# def count(results):
#     return len(results[0].boxes) if results else 0

# # Streaming with Detection
# def camera_stream():
#     while True:
#         img_resp = requests.get(url)
#         img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8)
#         img = cv2.imdecode(img_arr, -1)
#         img = imutils.resize(img, width=1000)

#         # Predictions
#         result_count = predict_inventory(img)  # SKU for total count
#         result_loreal = predict_loreal(img)
#         result_egg = predict_egg(img)
#         result_drinks = predict_drinks(img)
#         result_cold_drinks = predict_cold_drinks(img)
#         result_toothpaste = predict_toothpaste(img)
#         result_grocery = predict_grocery(img)

#         # Total count from SKU model only
#         total_count = count(result_count)

#         # Draw bounding boxes with custom labels and colors
#         # SKU (Green, no specific label unless provided)
#         for r in result_count:
#             for box in r.boxes:
#                 x1, y1, x2, y2 = map(int, box.xyxy[0])
#                 label = r.names[int(box.cls)] if r.names else "SKU"
#                 cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
#                 cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

#         # Loreal (Blue, retain detailed labels)
#         for r in result_loreal:
#             for box in r.boxes:
#                 x1, y1, x2, y2 = map(int, box.xyxy[0])
#                 label = r.names[int(box.cls)]
#                 cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
#                 cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

#         # Eggs (Blue, label as "Egg")
#         for r in result_egg:
#             for box in r.boxes:
#                 x1, y1, x2, y2 = map(int, box.xyxy[0])
#                 label = "Egg"
#                 cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
#                 cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

#         # Drinks (Orange, label as "Colddrink" assuming it's a class)
#         for r in result_drinks:
#             for box in r.boxes:
#                 x1, y1, x2, y2 = map(int, box.xyxy[0])
#                 label = "Colddrink" if "colddrink" in r.names[int(box.cls)].lower() else r.names[int(box.cls)]
#                 cv2.rectangle(img, (x1, y1), (x2, y2), (255, 165, 0), 2)
#                 cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 165, 0), 2)

#         # New Cold Drinks (Purple, label based on dataset classes)
#         for r in result_cold_drinks:
#             for box in r.boxes:
#                 x1, y1, x2, y2 = map(int, box.xyxy[0])
#                 cls_id = int(box.cls)
#                 if 0 <= cls_id < 6:
#                     label = ['Coca Cola', 'Sprite', 'Pepsi', 'Mountain Dew', '7UP', 'Fanta'][cls_id]
#                 else:
#                     label = "Colddrink"
#                 cv2.rectangle(img, (x1, y1), (x2, y2), (128, 0, 128), 2)
#                 cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 0, 128), 2)

#         # Toothpaste (Teal, label based on dataset)
#         for r in result_toothpaste:
#             for box in r.boxes:
#                 x1, y1, x2, y2 = map(int, box.xyxy[0])
#                 cls_id = int(box.cls)
#                 if 0 <= cls_id < 1:
#                     label = ['toothpaste'][cls_id]
#                 else:
#                     label = "Item"
#                 cv2.rectangle(img, (x1, y1), (x2, y2), (0, 128, 128), 2)
#                 cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 128, 128), 2)

#         # Grocery (Olive, label based on dataset - update with actual classes from data.yaml)
#         for r in result_grocery:
#             for box in r.boxes:
#                 x1, y1, x2, y2 = map(int, box.xyxy[0])
#                 cls_id = int(box.cls)
#                 if 0 <= cls_id < 10:  # Placeholder for 10 classes, adjust based on data.yaml
#                     label = ['item1', 'item2', 'item3', 'item4', 'item5', 'item6', 'item7', 'item8', 'item9', 'item10'][cls_id]
#                 else:
#                     label = "Grocery"
#                 cv2.rectangle(img, (x1, y1), (x2, y2), (128, 128, 0), 2)  # Olive color
#                 cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 0), 2)

#         # Text overlay (only total count from SKU on the left side)
#         cv2.putText(img, f"{total_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

#         _, buffer = cv2.imencode('.jpg', img)
#         frame = buffer.tobytes()
#         yield (b'--frame\r\n'
#                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/video_feed')
# def video_feed():
#     return Response(camera_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')

# if __name__ == "__main__":
#     app.run(debug=True, port=5000)

import cv2
import numpy as np
import imutils
import requests
from flask import Flask, render_template, Response
from ultralytics import YOLO

app = Flask(__name__, template_folder='./templates', static_folder='./static')


# Load Models with forward slashes
count_model = YOLO('C:/Users/ANISH SINHA/OneDrive/Desktop/ALL/FFFFFFFFFF/New folder/mc rahul/models/sku3/weights/best.pt')  # SKU (for total count)
loreal_model = YOLO('C:/Users/ANISH SINHA/OneDrive/Desktop/ALL/FFFFFFFFFF/New folder/mc rahul/models/loreal/weights/best.pt')  # Loreal
egg_model = YOLO('C:/Users/ANISH SINHA/OneDrive/Desktop/ALL/FFFFFFFFFF/New folder/mc rahul/models/egg_detection3/weights/best.pt')  # Egg Detection
drinks_model = YOLO('C:/Users/ANISH SINHA/OneDrive/Desktop/ALL/FFFFFFFFFF/New folder/mc rahul/models/drinks2/weights/best.pt')  # Drinks Detection
cold_drinks_model = YOLO('C:/Users/ANISH SINHA/OneDrive/Desktop/ALL/FFFFFFFFFF/New folder/mc rahul/models/cold_drinks_detection4/weights/best.pt')  # Cold Drinks Detection
toothpaste_model = YOLO('C:/Users/ANISH SINHA/OneDrive/Desktop/ALL/FFFFFFFFFF/New folder/mc rahul/models/toothpaste_detection/weights/best.pt')  # Toothpaste Detection
# grocery_model = YOLO('C:/Users/ANISH SINHA/OneDrive/Desktop/New folder/InventoryTallyProject/scripts/models/grocery_detection2/weights/best.pt')  # Grocery Detection

url = "http://192.168.34.33:8080/shot.jpg"

# Prediction Functions
def predict_inventory(frame):
    return count_model.predict(frame, conf=0.10)

def predict_loreal(frame):
    return loreal_model.predict(frame, conf=0.7)

def predict_egg(frame):
    return egg_model.predict(frame, conf=0.65)

def predict_drinks(frame):
    return drinks_model.predict(frame, conf=0.30)

def predict_cold_drinks(frame):
    return cold_drinks_model.predict(frame, conf=0.30)

def predict_toothpaste(frame):
    return toothpaste_model.predict(frame, conf=0.45)

#def predict_grocery(frame):
#    return grocery_model.predict(frame, conf=0.5)

def count(results):
    return len(results[0].boxes) if results else 0

# Streaming with Detection
def camera_stream():
    while True:
        img_resp = requests.get(url)
        img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8)
        img = cv2.imdecode(img_arr, -1)
        img = imutils.resize(img, width=1000)

        # Predictions
        result_count = predict_inventory(img)  # SKU for total count
        result_loreal = predict_loreal(img)
        result_egg = predict_egg(img)
        result_drinks = predict_drinks(img)
        result_cold_drinks = predict_cold_drinks(img)
        result_toothpaste = predict_toothpaste(img)
        #result_grocery = predict_grocery(img)

        # Total count from SKU model only
        total_count = count(result_count)

        # Draw bounding boxes with custom labels and colors
        # SKU (Green, no specific label unless provided)
        for r in result_count:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label = r.names[int(box.cls)] if r.names else "SKU"
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Loreal (Blue, retain detailed labels)
        for r in result_loreal:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label = r.names[int(box.cls)]
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # Eggs (Blue, label as "Egg")
        for r in result_egg:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label = "Egg"
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # Drinks (Orange, label as "Colddrink" assuming it's a class)
        for r in result_drinks:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label = "Colddrink" if "colddrink" in r.names[int(box.cls)].lower() else r.names[int(box.cls)]
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 165, 0), 2)
                cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 165, 0), 2)

        # New Cold Drinks (Purple, label based on dataset classes)
        for r in result_cold_drinks:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls_id = int(box.cls)
                if 0 <= cls_id < 6:
                    label = ['Coca Cola', 'Sprite', 'Pepsi', 'Mountain Dew', '7UP', 'Fanta'][cls_id]
                else:
                    label = "Colddrink"
                cv2.rectangle(img, (x1, y1), (x2, y2), (128, 0, 128), 2)
                cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 0, 128), 2)

        # Toothpaste (Teal, label based on dataset)
        for r in result_toothpaste:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls_id = int(box.cls)
                if 0 <= cls_id < 1:
                    label = ['toothpaste'][cls_id]
                else:
                    label = "Item"
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 128, 128), 2)
                cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 128, 128), 2)

        

        # Text overlay (only total count from SKU on the left side)
        cv2.putText(img, f"{total_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        _, buffer = cv2.imencode('.jpg', img)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(camera_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True, port=5000)
    