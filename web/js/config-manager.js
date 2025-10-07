/**
 * Configuration Management Module
 * Handles configuration saving, loading, and validation
 */

const DEFAULT_CONFIG = {
  numVehicles: 3,
  maxTripDuration: 480, // in minutes
  waitTime: 10, // in minutes
  maxEpochs: 20,
  mutationRate: 5, // in percentage
  maxNoImprovement: 50,
  vehicleMaxPoints: 8, // max points per vehicle
  companyAddress: { lat: -23.55, lng: -46.63 }, // Default to São Paulo center
}

/**
 * Get current configuration from form fields with localStorage fallback
 * @returns {Object} Configuration object
 */
function getConfig() {
  // Try to get from localStorage first
  let savedConfig = {};
  try {
    const stored = localStorage.getItem("routingConfig");
    if (stored) {
      savedConfig = JSON.parse(stored);
    }
  } catch (e) {
    console.warn("Error loading config from localStorage:", e);
  }

  // Get current form values, falling back to localStorage values
  const companyAddressValue = document.getElementById("company-address").value;
  let companyAddress = null;
  if (companyAddressValue) {
    const coords = companyAddressValue.split(',').map(c => parseFloat(c.trim()));
    if (coords.length === 2 && !isNaN(coords[0]) && !isNaN(coords[1])) {
      companyAddress = { lat: coords[0], lng: coords[1] };
    }
  }

  return {
    numVehicles:
      savedConfig.numVehicles || DEFAULT_CONFIG.numVehicles,
    maxTripDuration:
      savedConfig.maxTripDuration || DEFAULT_CONFIG.maxTripDuration,
    waitTime:
      savedConfig.waitTime || DEFAULT_CONFIG.waitTime,
    maxEpochs:
      savedConfig.maxEpochs || DEFAULT_CONFIG.maxEpochs,
    mutationRate:
      savedConfig.mutationRate || DEFAULT_CONFIG.mutationRate,
    maxNoImprovement:
      savedConfig.maxNoImprovement || DEFAULT_CONFIG.maxNoImprovement,
    vehicleMaxPoints:
      savedConfig.vehicleMaxPoints || DEFAULT_CONFIG.vehicleMaxPoints,
    companyAddress: companyAddress || savedConfig.companyAddress || DEFAULT_CONFIG.companyAddress,
  };
}

/**
 * Fill form fields with configuration values
 */
function fillFieldsFromConfig() {
  const config = getConfig();

  document.getElementById("num-vehicles").value = config.numVehicles;
  document.getElementById("max-trip-duration").value = config.maxTripDuration;
  document.getElementById("wait-time").value = config.waitTime;
  document.getElementById("max-epochs").value = config.maxEpochs;
  document.getElementById("mutation-rate").value = config.mutationRate;
  document.getElementById("max-no-improvement").value = config.maxNoImprovement;
  document.getElementById("vehicle-max-points").value = config.vehicleMaxPoints;

  // Handle company address
  const addressToUse = config.companyAddress || DEFAULT_CONFIG.companyAddress;
  if (addressToUse) {
    document.getElementById("company-address").value = `${addressToUse.lat}, ${addressToUse.lng}`;
    setCompanyAddress(addressToUse.lat, addressToUse.lng, false);
  }
}

/**
 * Save configuration with validation
 */
function saveConfig() {
  const fields = [
    { id: "num-vehicles", min: 1 },
    { id: "max-trip-duration", min: 1 },
    { id: "wait-time", min: 0 },
    { id: "max-epochs", min: 1 },
    { id: "mutation-rate", min: 0, max: 100 },
    { id: "max-no-improvement", min: 1 },
    { id: "vehicle-max-points", min: 1 },
  ];

  let isValid = true;

  fields.forEach((field) => {
    const input = document.getElementById(field.id);
    const error = document.getElementById(`error-${field.id}`);
    const value = parseFloat(input.value);

    input.classList.remove("error");
    error.classList.remove("show");

    if (!input.value.trim() || isNaN(value)) {
      input.classList.add("error");
      error.classList.add("show");
      isValid = false;
    } else if (field.min !== undefined && value < field.min) {
      input.classList.add("error");
      error.classList.add("show");
      isValid = false;
    } else if (field.max !== undefined && value > field.max) {
      input.classList.add("error");
      error.classList.add("show");
      isValid = false;
    }
  });

  // Validate company address
  const companyAddressInput = document.getElementById("company-address");
  const companyAddressError = document.getElementById("error-company-address");
  companyAddressInput.classList.remove("error");
  companyAddressError.classList.remove("show");

  if (!companyAddressInput.value.trim()) {
    companyAddressInput.classList.add("error");
    companyAddressError.classList.add("show");
    isValid = false;
  } else {
    const coords = companyAddressInput.value.split(',').map(c => parseFloat(c.trim()));
    if (coords.length !== 2 || isNaN(coords[0]) || isNaN(coords[1])) {
      companyAddressInput.classList.add("error");
      companyAddressError.classList.add("show");
      isValid = false;
    }
  }

  if (isValid) {
    alert("Configuration saved successfully!");
    // Save to localStorage
    localStorage.setItem("routingConfig", JSON.stringify(getConfig()));
  }
}

/**
 * Load saved configuration on page load
 */
function loadSavedConfig() {
  const savedConfig = localStorage.getItem("routingConfig");
  let config = DEFAULT_CONFIG; // Start with default config

  if (savedConfig) {
    try {
      const parsed = JSON.parse(savedConfig);
      // Merge saved config with defaults
      config = { ...DEFAULT_CONFIG, ...parsed };
    } catch (e) {
      console.warn("Error parsing saved config, using defaults:", e);
    }
  }

  // Set form values with config (which now includes defaults)
  document.getElementById("num-vehicles").value = config.numVehicles;
  document.getElementById("max-trip-duration").value = config.maxTripDuration;
  document.getElementById("wait-time").value = config.waitTime;
  document.getElementById("max-epochs").value = config.maxEpochs;
  document.getElementById("mutation-rate").value = config.mutationRate;
  document.getElementById("max-no-improvement").value = config.maxNoImprovement;

  if (config.companyAddress) {
    document.getElementById("company-address").value = `${config.companyAddress.lat}, ${config.companyAddress.lng}`;
    setCompanyAddress(config.companyAddress.lat, config.companyAddress.lng, false);
  }
}

// Export functions to global scope for compatibility
window.getConfig = getConfig;
window.fillFieldsFromConfig = fillFieldsFromConfig;
window.saveConfig = saveConfig;
window.loadSavedConfig = loadSavedConfig;