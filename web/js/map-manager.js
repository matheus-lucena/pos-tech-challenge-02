/**
 * Map Management Module
 * Handles map initialization and interactions
 */

let map;
let companyAddressMarker;
let companyAddressMode = false;

/**
 * Initialize the Leaflet map
 */
function initMap() {
  map = L.map("map").setView([-23.55, -46.63], 12);

  L.tileLayer(
    "https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png",
    {
      attribution:
        '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>',
      subdomains: "abcd",
      maxZoom: 20,
    }
  ).addTo(map);

  // Click to add point or set company address
  map.on("click", function (e) {
    if (
      document.getElementById("routing-tab").classList.contains("active")
    ) {
      addPoint(e.latlng.lat, e.latlng.lng);
    } else if (companyAddressMode && 
               document.getElementById("config-tab").classList.contains("active")) {
      setCompanyAddress(e.latlng.lat, e.latlng.lng);
    }
  });

  // Set default company address
  setCompanyAddress(-23.55, -46.63, false);
}

/**
 * Set map cursor style based on active tab
 * @param {string} tabName - The active tab name
 */
function setMapCursor(tabName) {
  if (tabName === "routing" || (tabName === "config" && companyAddressMode)) {
    map.getContainer().classList.remove("normal-cursor");
  } else {
    map.getContainer().classList.add("normal-cursor");
  }
}

/**
 * Enable company address selection mode
 */
function setCompanyAddressMode() {
  companyAddressMode = true;
  setMapCursor("config");
  
  // Visual feedback
  document.getElementById("company-address").placeholder = "Click on map to select company address...";
  
  // Update button appearance
  const button = document.getElementById('company-address-btn');
  button.innerHTML = '<i class="fas fa-hand-pointer"></i> Click on Map';
  button.style.backgroundColor = '#eab308';
  button.style.color = '#0a0a0a';
}

/**
 * Set company address coordinates
 * @param {number} lat - Latitude
 * @param {number} lng - Longitude  
 * @param {boolean} updateConfig - Whether to update the config
 */
function setCompanyAddress(lat, lng, updateConfig = true) {
  // Remove existing marker
  if (companyAddressMarker) {
    map.removeLayer(companyAddressMarker);
  }

  // Add new marker
  companyAddressMarker = L.marker([lat, lng], {
    icon: L.divIcon({
      className: 'company-marker',
      html: '<i class="fas fa-building" style="color: #ffffff; font-size: 20px;"></i>',
      iconSize: [25, 25],
      iconAnchor: [12.5, 12.5]
    })
  }).addTo(map);

  // Update input field
  document.getElementById("company-address").value = `${lat.toFixed(6)}, ${lng.toFixed(6)}`;
  
  if (updateConfig) {
    // Exit selection mode
    companyAddressMode = false;
    setMapCursor("config");
    
    // Reset button
    const button = document.getElementById('company-address-btn');
    button.innerHTML = '<i class="fas fa-map-marker-alt"></i> Select on Map';
    button.style.backgroundColor = '#22c55e';
    button.style.color = '#0a0a0a';
    
    // Reset placeholder
    document.getElementById("company-address").placeholder = "-23.55, -46.63";
  }
}

// Export functions to global scope for compatibility
window.initMap = initMap;
window.setMapCursor = setMapCursor;
window.setCompanyAddressMode = setCompanyAddressMode;
window.setCompanyAddress = setCompanyAddress;