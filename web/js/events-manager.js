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
      showVehiclePanel(data.vehicle_data);
    }
  };
  
  eventSource.onerror = function(error) {
    console.warn('EventSource error:', error);
  };
}

/**
 * Show the vehicle panel with vehicle information
 * @param {Array} vehicleData - Array of vehicle data
 */
function showVehiclePanel(vehicleData) {
  const panel = document.getElementById('vehicle-panel');
  const vehicleList = document.getElementById('vehicle-list');
 
  console.log({vehicleData})

  // Clear existing content
  vehicleList.innerHTML = '';
  
  // Create vehicle items using real data from genetic algorithm
  vehicleData.forEach((vehicle, index) => {
    const color = getVehicleColor(index);
    const vehicleItem = createVehicleItem(vehicle, color, {
      distance: vehicle.distance || 0,
      duration: vehicle.duration || 0
    });
    vehicleList.appendChild(vehicleItem);
  });
  
  // Show the panel
  panel.classList.remove('hidden');
}

/**
 * Create a vehicle item element
 * @param {Object} vehicle - Vehicle data
 * @param {string} color - Vehicle color
 * @param {Object} routeData - Route data with distance and duration
 * @returns {HTMLElement} Vehicle item element
 */
function createVehicleItem(vehicle, color, routeData = null) {
  const item = document.createElement('div');
  item.className = 'vehicle-item';
  
  // Use real route data or placeholders
  const distance = routeData ? routeData.distance : Math.round(Math.random() * 50 + 10);
  const duration = routeData ? routeData.duration : Math.round(Math.random() * 120 + 30);
  const stops = vehicle.points || 0;
  
  item.innerHTML = `
    <div class="vehicle-header">
      <div class="vehicle-color" style="background-color: ${color}"></div>
      <div class="vehicle-name">Vehicle ${vehicle.vehicle_id}</div>
    </div>
    <div class="vehicle-stats">
      <div class="vehicle-stat">
        <i class="fas fa-route"></i>
        <span>Distance</span>
        <span class="vehicle-stat-value">${distance} km</span>
      </div>
      <div class="vehicle-stat">
        <i class="fas fa-clock"></i>
        <span>Duration</span>
        <span class="vehicle-stat-value">${duration} min</span>
      </div>
      <div class="vehicle-stat">
        <i class="fas fa-map-marker-alt"></i>
        <span>Stops</span>
        <span class="vehicle-stat-value">${stops}</span>
      </div>
    </div>
  `;
  
  return item;
}

/**
 * Hide the vehicle panel
 */
function hideVehiclePanel() {
  const panel = document.getElementById('vehicle-panel');
  panel.classList.add('hidden');
}

/**
 * Initialize vehicle panel event listeners
 */
function initVehiclePanel() {
  const closeBtn = document.getElementById('vehicle-panel-close');
  closeBtn.addEventListener('click', hideVehiclePanel);
}

window.initEventSource = initEventSource;
window.showVehiclePanel = showVehiclePanel;
window.hideVehiclePanel = hideVehiclePanel;
window.initVehiclePanel = initVehiclePanel;