/**
 * Vehicle Colors Configuration
 * Central location for all vehicle colors used across the application
 */

// Colors for different vehicles (20 distinct colors)
const VEHICLE_COLORS = [
  '#ff4444', '#44ff44', '#4444ff', '#ffff44',
  '#ff44ff', '#44ffff', '#ff8844', '#8844ff',
  '#88ff44', '#ff4488', '#4488ff', '#ffaa44',
  '#aa44ff', '#44ffaa', '#ff6666', '#66ff66',
  '#6666ff', '#ffcc44', '#cc44ff', '#44ccff'
];

/**
 * Get vehicle color by index
 * @param {number} index - Vehicle index
 * @returns {string} Vehicle color
 */
function getVehicleColor(index) {
  return VEHICLE_COLORS[index % VEHICLE_COLORS.length];
}

/**
 * Get all vehicle colors
 * @returns {Array} Array of vehicle colors
 */
function getVehicleColors() {
  return VEHICLE_COLORS;
}

// Export functions
window.getVehicleColor = getVehicleColor;
window.getVehicleColors = getVehicleColors;
window.VEHICLE_COLORS = VEHICLE_COLORS;