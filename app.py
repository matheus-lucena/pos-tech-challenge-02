from flask import Flask, send_file, request
from flask_socketio import SocketIO
import json
import threading
from vrp.main import run_vrp

app = Flask(__name__, static_folder='web', static_url_path='')
socketio = SocketIO(app, cors_allowed_origins="*")

def send_training_event(epoch, **kwargs):
    event_data = {
        'status': 'training',
        'epoch': epoch,
        **kwargs
    }
    socketio.emit('training_update', event_data)

@app.route('/')
def index():
    return send_file('web/index.html')

@socketio.on('connect')
def handle_connect():
    print('Client connected')

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

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
            generate_json=True,
            epoch_callback=send_training_event,
        )
        # Send finished message when training completes
        socketio.emit('training_update', {"status": "finished"})

    training_thread = threading.Thread(
      target=run_with_completion,
      daemon=True
    )
    training_thread.start()
    
    return {"result": "Route calculation started"}, 202

if __name__ == '__main__':
    print("Flask-SocketIO server starting on http://localhost:5002")
    socketio.run(app, debug=True, port=5002)
