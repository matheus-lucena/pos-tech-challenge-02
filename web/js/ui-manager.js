/**
 * UI Management Module
 * Handles tab switching and UI interactions
 */

/**
 * Initialize tab switching functionality
 */
function initTabSwitching() {
  document.querySelectorAll(".tab-button").forEach((button) => {
    button.addEventListener("click", function () {
      const tabName = this.getAttribute("data-tab");
      switchTab(tabName);
    });
  });
}

/**
 * Switch between tabs
 * @param {string} tabName - Name of the tab to switch to
 */
function switchTab(tabName) {
  // Remove active class from all tabs and content
  document
    .querySelectorAll(".tab-button")
    .forEach((b) => b.classList.remove("active"));
  document
    .querySelectorAll(".tab-content")
    .forEach((c) => c.classList.remove("active"));

  // Add active class to selected tab and corresponding content
  document.querySelector(`[data-tab="${tabName}"]`).classList.add("active");
  document.getElementById(`${tabName}-tab`).classList.add("active");

  // Change cursor based on tab
  setMapCursor(tabName);

  // Fill fields with config when Config tab is focused
  if (tabName === "config") {
    fillFieldsFromConfig();
  }
  
  // Load recents when switching to recents tab
  if (tabName === 'recents') {
    loadRecents();
  }
}

/**
 * Load and display recent routes
 */
function loadRecents() {
  const recentsList = document.getElementById("recents-list");
  
  try {
    const recents = JSON.parse(localStorage.getItem("recentRoutes") || "[]");
    
    if (recents.length === 0) {
      recentsList.innerHTML = '<p style="color: var(--color-text-muted); text-align: center; padding: 20px">Nenhum cálculo recente ainda</p>';
      return;
    }
    
    recentsList.innerHTML = recents.map(route => `
      <div class="recents-item" data-route-id="${route.id}"
           onmouseenter="showRoutePreview(${route.id})"
           onmouseleave="hideRoutePreview()">
        <div class="recents-item-content" onclick="loadRecentRoute(${route.id})">
          <div class="recents-item-title">
            <i class="fas fa-route"></i> 
            Rota com ${route.pointsCount} pontos
          </div>
          <div class="recents-item-meta">
            <i class="fas fa-clock"></i> ${route.dateFormatted}
          </div>
        </div>
        <div class="recents-item-actions">
          <button class="copy-btn" onclick="event.stopPropagation(); copyRoutePoints(${route.id})" title="Copy points as text">
            <i class="fas fa-copy"></i>
          </button>
        </div>
      </div>
    `).join('');
  } catch (error) {
    console.warn("Error loading recents:", error);
    recentsList.innerHTML = '<p style="color: var(--color-error); text-align: center; padding: 20px">Erro ao carregar histórico</p>';
  }
}

let previewMarkers = [];

/**
 * Show route preview on map when hovering over recent item
 * @param {number} routeId - ID of the route to preview
 */
function showRoutePreview(routeId) {
  try {
    const recents = JSON.parse(localStorage.getItem("recentRoutes") || "[]");
    const route = recents.find(r => r.id === routeId);
    
    if (route && route.points) {
      // Clear any existing preview markers
      hideRoutePreview();
      
      // Add preview markers for this route
      route.points.forEach((point, index) => {
        const icon = L.divIcon({
          className: "custom-div-icon",
          html: `<div style="background-color: var(--color-warning); color: var(--color-bg-primary); border-radius: 50%; width: 24px; height: 24px; display: flex; align-items: center; justify-content: center; font-weight: bold; border: 2px solid var(--color-bg-primary); opacity: 0.8; box-shadow: 0 0 8px rgba(251, 191, 36, 0.6);">${index + 1}</div>`,
          iconSize: [24, 24],
          iconAnchor: [12, 12],
        });

        const marker = L.marker([point.lat, point.lng], { icon: icon })
          .bindTooltip(`Preview Point ${index + 1}`, { permanent: false })
          .addTo(map);

        previewMarkers.push(marker);
      });
    }
  } catch (error) {
    console.warn("Error showing route preview:", error);
  }
}

/**
 * Hide route preview markers
 */
function hideRoutePreview() {
  previewMarkers.forEach(marker => {
    map.removeLayer(marker);
  });
  previewMarkers = [];
}

/**
 * Initialize all UI components
 */
function initUI() {
  initTabSwitching();
}

// Export functions to global scope for compatibility
window.switchTab = switchTab;
window.loadRecents = loadRecents;
window.initUI = initUI;