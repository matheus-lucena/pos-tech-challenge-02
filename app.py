from flask import Flask, Response, send_file, request
import time
import json
from queue import Queue
import threading
from vrp.main import run_vrp

app = Flask(__name__, static_folder='web', static_url_path='')

event_queue = Queue(maxsize=1000)

def send_training_event(epoch, **kwargs):
    event_data = {
        'epoch': epoch,
        **kwargs
    }
    event_queue.put(event_data)

@app.route('/')
def index():
    return send_file('web/index.html')

@app.route('/stream')
def stream():
    def event_stream():
        while True:
            # Get event from queue (blocks until available)
            if not event_queue.empty():
                event_data = event_queue.get()
                yield f"data: {json.dumps(event_data)}\n\n"
            time.sleep(0.1)  # Small delay to prevent busy waiting
    
    return Response(event_stream(), mimetype='text/event-stream')

@app.route('/calculate-route', methods=['POST'])
def calculate_route():
    data = request.json
    points = data.get("points", [])
    config = data.get("config", {})
    
    if not points:
        return {"error": "No points provided"}, 400

    company_address = config['companyAddress']
    company_address = (company_address['lat'], company_address['lng'])
    points_list = [(p['lat'], p['lng']) for p in points]
    
    points_list.insert(0, company_address)
    
    training_thread = threading.Thread(
      target=run_vrp,
      kwargs={
        'points': points_list,
        'max_epochs': config['maxEpochs'],
        'num_vehicles': config['numVehicles'],
        'vehicle_max_points': config['vehicleMaxPoints'],
        'epoch_callback': send_training_event
      },
      daemon=True
    )
    training_thread.start()
    
    return {"result": "Route calculation started"}, 202

if __name__ == '__main__':
    # Run Flask server
    app.run(debug=True, threaded=True, port=5002)
