from flask import Flask, redirect, url_for, request, render_template

app = Flask(__name__)

@app.route('/', methods=['GET'])
def serve_index():
    return render_template('./index.html')

#@app.route('/images', methods=['POST'])
#def annotate_images():
    #do something

#@app.route('/retrain', methods=['POST'])
#def fine_tune():
    # tune model with images


if __name__ == '__main__':
    app.run()