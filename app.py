from flask import Flask, Response, send_file, request
import time
import json
from queue import Queue

app = Flask(__name__, static_folder='web', static_url_path='')

event_queue = Queue()

def send_training_event(epoch, loss, accuracy, **kwargs):
    """
    Send a training event to all connected SSE clients.
    
    Args:
        epoch: Current epoch number
        loss: Current loss value
        accuracy: Current accuracy value
        **kwargs: Any additional metrics you want to send
    """
    print(f"Epoch {epoch}: loss={loss}, accuracy={accuracy}, extras={kwargs}")
    event_data = {
        'epoch': epoch,
        'loss': round(loss, 4),
        'accuracy': round(accuracy, 4),
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
            time.sleep(0.01)  # Small delay to prevent busy waiting
    
    return Response(event_stream(), mimetype='text/event-stream')

@app.route('/calculate-route', methods=['POST'])
def calculate_route():
    data = request.json
    points = data.get("points", [])
    config = data.get("config", {})

    if not points:
        return {"error": "No points provided"}, 400

    return {"result": "Route calculated successfully"}

if __name__ == '__main__':
    # Run Flask server
    app.run(debug=True, threaded=True, port=5002)
