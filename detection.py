from imageai.Detection import ObjectDetection


detector = ObjectDetection()

# Path from our input image, output image, and model
model_path = "./models/yolo-tiny.h5"
input_path = "./input/car.jpg"
output_path = "./output/newimage.jpg"

# Load our model
detector.setModelTypeAsTinyYOLOv3()
# This function accepts a string which contains the path to the pre-trained model
detector.setModelPath(model_path)
# Loads the model from the above path using the setModelPath()
detector.loadModel()

detection = detector.detectObjectsFromImage(
    input_image=input_path, output_image_path=output_path)

for eachItem in detection:
    print(eachItem["name"], " : ", eachItem["percentage_probability"])
