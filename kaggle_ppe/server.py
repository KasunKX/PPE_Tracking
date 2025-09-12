from flask import Flask, render_template, Response,
import json
from flask_cors import CORS


app = Flask(__name__)


@app.route('/counts')
def counts():
    with open("ppe_counts.json", "r") as f:
        data = json.load(f)
        
    return {"data": data}

if __name__ == "__main__":
    # Host 0.0.0.0 makes it accessible on LAN
    app.run(host='0.0.0.0', port=5001)
