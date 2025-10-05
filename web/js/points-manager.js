/**
 * Points Management Module
 * Handles point addition, removal, and display
 */

let points = [];
let markers = [];
const MAX_POINTS = 400;

/**
 * Add a point to the map and points list
 * @param {number} lat - Latitude
 * @param {number} lng - Longitude
 */
function addPoint(lat, lng) {
  if (points.length >= MAX_POINTS) {
    alert(`Maximum ${MAX_POINTS} points allowed`);
    return;
  }

  const point = {
    lat: parseFloat(lat.toFixed(6)),
    lng: parseFloat(lng.toFixed(6)),
  };
  points.push(point);

  // Add marker
  const icon = L.divIcon({
    className: "custom-div-icon",
    html: `<div style="background-color: #22c55e; color: #0a0a0a; border-radius: 50%; width: 32px; height: 32px; display: flex; align-items: center; justify-content: center; font-weight: bold; border: 3px solid #0a0a0a; box-shadow: 0 0 10px rgba(34, 197, 94, 0.5);">${points.length}</div>`,
    iconSize: [32, 32],
    iconAnchor: [16, 16],
  });

  const marker = L.marker([lat, lng], { icon: icon })
    .bindPopup(
      `<strong>Point ${points.length}</strong><br>${lat.toFixed(
        6
      )}, ${lng.toFixed(6)}`
    )
    .addTo(map);

  markers.push(marker);
  updatePointsList();
}

/**
 * Add point manually from coordinates input
 */
function addManualPoint() {
  const input = document.getElementById("manual-coords");
  const coords = input.value.trim().split(",");

  if (coords.length !== 2) {
    alert("Please enter coordinates in format: lat, long");
    return;
  }

  const lat = parseFloat(coords[0].trim());
  const lng = parseFloat(coords[1].trim());

  if (
    isNaN(lat) ||
    isNaN(lng) ||
    lat < -90 ||
    lat > 90 ||
    lng < -180 ||
    lng > 180
  ) {
    alert(
      "Invalid coordinates. Latitude must be between -90 and 90, longitude between -180 and 180"
    );
    return;
  }

  addPoint(lat, lng);
  input.value = "";
}

/**
 * Remove a point from the map and points list
 * @param {number} index - Index of the point to remove
 */
function removePoint(index) {
  points.splice(index, 1);
  map.removeLayer(markers[index]);
  markers.splice(index, 1);

  // Update remaining markers
  markers.forEach((marker, i) => {
    marker.remove();
  });
  markers = [];

  points.forEach((point, i) => {
    const icon = L.divIcon({
      className: "custom-div-icon",
      html: `<div style="background-color: #22c55e; color: #0a0a0a; border-radius: 50%; width: 32px; height: 32px; display: flex; align-items: center; justify-content: center; font-weight: bold; border: 3px solid #0a0a0a; box-shadow: 0 0 10px rgba(34, 197, 94, 0.5);">${i + 1
        }</div>`,
      iconSize: [32, 32],
      iconAnchor: [16, 16],
    });

    const marker = L.marker([point.lat, point.lng], { icon: icon })
      .bindPopup(
        `<strong>Point ${i + 1}</strong><br>${point.lat.toFixed(
          6
        )}, ${point.lng.toFixed(6)}`
      )
      .addTo(map);

    markers.push(marker);
  });

  updatePointsList();
}

/**
 * Update the points list display
 */
function updatePointsList() {
  const list = document.getElementById("points-list");

  if (points.length === 0) {
    list.innerHTML =
      '<p style="color: #737373; text-align: center; padding: 20px;">No points added yet</p>';
    return;
  }

  list.innerHTML = points
    .map(
      (point, i) => `
          <div class="point-item">
              <div class="point-number">${i + 1}</div>
              <div class="point-coords">${point.lat.toFixed(
        6
      )}, ${point.lng.toFixed(6)}</div>
              <button class="point-remove" onclick="removePoint(${i})">
                  <i class="fas fa-times"></i>
              </button>
          </div>
      `
    )
    .join("");
}

/**
 * Calculate route using the VRP algorithm
 */
async function calculateRoute() {
  if (points.length === 0) {
    alert("Please add at least one point");
    return;
  }

  const btn = document.getElementById("calculate-btn");
  btn.disabled = true;
  btn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Calculating...';

  try {
    const response = await fetch("/calculate-route", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        points: points,
        config: getConfig(),
      }),
    });

    const result = await response.json();

    if (response.ok) {
      // alert("Route calculated successfully!");
      // console.log("Result:", result);

      // Save to recents
      saveToRecents(points);

      // Handle result here (draw route, show stats, etc.)
    } else {
      alert("Error: " + (result.error || "Failed to calculate route"));
    }
  } catch (error) {
    alert("Error connecting to server: " + error.message);
  } finally {
    btn.disabled = false;
    btn.innerHTML = '<i class="fas fa-calculator"></i> Calculate Route';
  }
}

/**
 * Save current route to recents in localStorage
 * @param {Array} routePoints - Array of points to save
 */
function saveToRecents(routePoints) {
  if (!routePoints || routePoints.length === 0) return;

  try {
    let recents = JSON.parse(localStorage.getItem("recentRoutes") || "[]");

    const newRoute = {
      id: Date.now(),
      points: [...routePoints],
      pointsCount: routePoints.length,
      date: new Date().toISOString(),
      dateFormatted: new Date().toLocaleString("pt-BR")
    };

    // Add to beginning of array
    recents.unshift(newRoute);

    // Keep only last 15 routes
    if (recents.length > 15) {
      recents = recents.slice(0, 15);
    }

    localStorage.setItem("recentRoutes", JSON.stringify(recents));
  } catch (error) {
    console.warn("Error saving to recents:", error);
  }
}

/**
 * Load a recent route and switch to routing tab
 * @param {number} routeId - ID of the route to load
 */
function loadRecentRoute(routeId) {
  try {
    const recents = JSON.parse(localStorage.getItem("recentRoutes") || "[]");
    const route = recents.find(r => r.id === routeId);

    if (route) {
      // Clear current points
      points = [];
      markers.forEach(marker => map.removeLayer(marker));
      markers = [];

      // Add route points
      route.points.forEach(point => {
        addPoint(point.lat, point.lng);
      });

      // Switch to routing tab
      switchTab('routing');
    }
  } catch (error) {
    console.warn("Error loading recent route:", error);
  }
}

// Export functions to global scope for compatibility
window.addPoint = addPoint;
window.addManualPoint = addManualPoint;
window.removePoint = removePoint;
window.updatePointsList = updatePointsList;
window.calculateRoute = calculateRoute;
window.saveToRecents = saveToRecents;
window.loadRecentRoute = loadRecentRoute;