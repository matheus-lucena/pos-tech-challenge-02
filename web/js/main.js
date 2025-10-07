/**
 * Main Application Entry Point
 * Initializes all modules and components
 */

/**
 * Initialize the entire application
 */
function initApp() {
  // Initialize map
  initMap();

  // Initialize UI components
  initUI();

  // Initialize points list
  updatePointsList();

  // Load saved configuration
  loadSavedConfig();

  // Initialize event source for real-time updates
  initEventSource();
  
  // Initialize vehicle panel
  initVehiclePanel();
}

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', initApp);
