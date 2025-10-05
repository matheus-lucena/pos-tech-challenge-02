/**
 * Initialize EventSource for real-time updates
 */
function initEventSource() {
  const eventSource = new EventSource('/stream');
  
  eventSource.onmessage = function(event) {
    const data = JSON.parse(event.data);
    console.log('Training update:', data);
    
    // Update UI with route visualization
    if (data.vehicle_data && Array.isArray(data.vehicle_data)) {
      updateRouteVisualization(data);
    }
  };
  
  eventSource.onerror = function(error) {
    console.warn('EventSource error:', error);
  };
}

window.initEventSource = initEventSource;