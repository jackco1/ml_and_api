from yolov3.yolo_detection_images import annotateImages
from flask import Flask, jsonify, request, render_template


app = Flask(__name__)

@app.route('/', methods=['GET'])
def serve_index():
    return render_template('./index.html')

@app.route('/images', methods=['POST'])
def detect():
    return jsonify(annotateImages())

#@app.route('/retrain', methods=['POST'])
#def fine_tune():
    # tune model with images


if __name__ == '__main__':
    app.run()