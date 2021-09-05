from yolov3.yolo_detection_images import annotateImages
from flask import Flask, jsonify, request, render_template


app = Flask(__name__)

#default page, probs wont make it to production but its just useful
#for testing
@app.route('/', methods=['GET'])
def serve_index():
    return render_template('./index.html')

# ideally should be able to send a bunch of images here at once
@app.route('/images', methods=['POST'])
def detect():
    return annotateImages()

#@app.route('/retrain', methods=['POST'])
#def fine_tune():
    # tune model with images


if __name__ == '__main__':
    app.run(debug=True)