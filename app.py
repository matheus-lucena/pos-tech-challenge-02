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
        'status': 'training',
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

    # Format address
    company_address = config['companyAddress']
    company_address = (company_address['lat'], company_address['lng'])
    points_list = [(p['lat'], p['lng']) for p in points]
    
    points_list.insert(0, company_address)
    
    # Format max trip duration
    max_trip_duration = config['maxTripDuration']
    max_trip_duration = max_trip_duration * 60  # Convert minutes to seconds
    
    # Format max wait time
    wait_time = config['waitTime']
    wait_time = wait_time * 60  # Convert minutes to seconds

    def run_with_completion():
        run_vrp(
            points=points_list,
            max_epochs=config['maxEpochs'],
            num_vehicles=config['numVehicles'],
            vehicle_max_points=config['vehicleMaxPoints'],
            max_trip_distance=config['maxTripDistance'],
            wait_time=wait_time,
            max_trip_duration=max_trip_duration,
            mutation_rate=config['mutationRate'] / 100.0,
            max_no_improvement=config['maxNoImprovement'],
            epoch_callback=send_training_event
        )
        # Send finished message when training completes
        event_queue.put({"status": "finished"})

    training_thread = threading.Thread(
      target=run_with_completion,
      daemon=True
    )
    training_thread.start()
    
    return {"result": "Route calculation started"}, 202

if __name__ == '__main__':
    # Run Flask server
    app.run(debug=True, threaded=True, port=5002)
