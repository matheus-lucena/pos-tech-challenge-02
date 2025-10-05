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

  // Initialize EventSource for real-time updates
  initEventSource();
}

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', initApp);
