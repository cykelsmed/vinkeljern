"""
Health dashboard for Vinkeljernet.

This module provides a web-based health dashboard for monitoring the status
of Vinkeljernet services in real-time. It visualizes data from the app_status
module and allows operators to see the health of all components at a glance.

Usage:
    # Start the dashboard in a separate thread
    dashboard = HealthDashboard()
    dashboard.start(port=8080)
    
    # Access the dashboard at http://localhost:8080
"""

import os
import json
import time
import asyncio
import logging
import threading
import webbrowser
from typing import Dict, List, Any, Optional, Set
from datetime import datetime
from html import escape

# Try to import Flask, or show clear instructions if not installed
try:
    from flask import Flask, render_template, jsonify, request, Response
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False
    logging.warning(
        "Flask is not installed. To use the health dashboard, "
        "install with: pip install flask"
    )

from app_status import get_app_status, VinkeljernetAppStatus, HealthStatus
from fault_tolerance import get_all_services_status

# Configure logging
logger = logging.getLogger("vinkeljernet.health_dashboard")

# Default HTML template for dashboard
DEFAULT_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Vinkeljernet Health Dashboard</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 20px;
            color: #333;
            background-color: #f5f5f5;
        }
        
        h1, h2, h3 {
            color: #2c3e50;
        }
        
        .dashboard {
            max-width: 1200px;
            margin: 0 auto;
        }
        
        .status-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
            background-color: white;
            padding: 15px;
            border-radius: 5px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        .status-indicator {
            display: flex;
            align-items: center;
        }
        
        .status-circle {
            width: 24px;
            height: 24px;
            border-radius: 50%;
            margin-right: 10px;
        }
        
        .status-circle.healthy {
            background-color: #2ecc71;
        }
        
        .status-circle.warning {
            background-color: #f39c12;
        }
        
        .status-circle.degraded {
            background-color: #e67e22;
        }
        
        .status-circle.error {
            background-color: #e74c3c;
        }
        
        .timestamp {
            color: #7f8c8d;
            font-size: 0.9em;
        }
        
        .services-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        
        .service-card {
            background-color: white;
            border-radius: 5px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            padding: 15px;
            transition: transform 0.2s;
        }
        
        .service-card:hover {
            transform: translateY(-5px);
        }
        
        .service-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 10px;
        }
        
        .service-name {
            font-weight: bold;
            font-size: 1.1em;
            color: #2c3e50;
        }
        
        .service-status {
            padding: 3px 8px;
            border-radius: 12px;
            font-size: 0.8em;
            font-weight: bold;
        }
        
        .service-status.healthy {
            background-color: #d5f5e3;
            color: #27ae60;
        }
        
        .service-status.degraded, .service-status.warning {
            background-color: #fef2d9;
            color: #d35400;
        }
        
        .service-status.error {
            background-color: #fadbd8;
            color: #c0392b;
        }
        
        .circuit-state {
            margin-top: 10px;
            font-size: 0.9em;
        }
        
        .circuit-state .open {
            color: #e74c3c;
            font-weight: bold;
        }
        
        .circuit-state .half-open {
            color: #f39c12;
            font-weight: bold;
        }
        
        .circuit-state .closed {
            color: #2ecc71;
        }
        
        .detail-row {
            display: flex;
            justify-content: space-between;
            margin-top: 5px;
            font-size: 0.9em;
            border-top: 1px solid #f0f0f0;
            padding-top: 5px;
        }
        
        .detail-label {
            color: #7f8c8d;
        }
        
        .issues-container {
            background-color: white;
            border-radius: 5px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            padding: 15px;
            margin-bottom: 30px;
        }
        
        .issue {
            padding: 10px;
            border-radius: 5px;
            margin-bottom: 10px;
        }
        
        .issue.error {
            background-color: #fadbd8;
        }
        
        .issue.warning {
            background-color: #fef2d9;
        }
        
        .actions {
            margin-top: 20px;
            display: flex;
            gap: 10px;
        }
        
        button {
            background-color: #3498db;
            color: white;
            border: none;
            padding: 8px 15px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 0.9em;
        }
        
        button:hover {
            background-color: #2980b9;
        }
        
        .loading {
            display: inline-block;
            width: 20px;
            height: 20px;
            border: 3px solid rgba(0,0,0,.1);
            border-radius: 50%;
            border-top-color: #3498db;
            animation: spin 1s ease-in-out infinite;
            margin-right: 10px;
        }
        
        @keyframes spin {
            to { transform: rotate(360deg); }
        }
        
        .history-container {
            margin-top: 30px;
            background-color: white;
            border-radius: 5px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            padding: 15px;
        }
        
        .history-item {
            display: flex;
            align-items: center;
            padding: 8px 0;
            border-bottom: 1px solid #f0f0f0;
        }
        
        .history-item:last-child {
            border-bottom: none;
        }

        /* Auto refresh toggle */
        .switch {
            position: relative;
            display: inline-block;
            width: 54px;
            height: 28px;
        }
        
        .switch input {
            opacity: 0;
            width: 0;
            height: 0;
        }
        
        .slider {
            position: absolute;
            cursor: pointer;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background-color: #ccc;
            transition: .4s;
            border-radius: 34px;
        }
        
        .slider:before {
            position: absolute;
            content: "";
            height: 20px;
            width: 20px;
            left: 4px;
            bottom: 4px;
            background-color: white;
            transition: .4s;
            border-radius: 50%;
        }
        
        input:checked + .slider {
            background-color: #2ecc71;
        }
        
        input:checked + .slider:before {
            transform: translateX(26px);
        }

        /* Tooltip styles */
        .tooltip {
            position: relative;
            display: inline-block;
        }

        .tooltip .tooltiptext {
            visibility: hidden;
            width: 250px;
            background-color: #555;
            color: #fff;
            text-align: left;
            border-radius: 6px;
            padding: 10px;
            position: absolute;
            z-index: 1;
            bottom: 125%;
            left: 50%;
            margin-left: -125px;
            opacity: 0;
            transition: opacity 0.3s;
            font-size: 0.8em;
            line-height: 1.4;
        }

        .tooltip .tooltiptext::after {
            content: "";
            position: absolute;
            top: 100%;
            left: 50%;
            margin-left: -5px;
            border-width: 5px;
            border-style: solid;
            border-color: #555 transparent transparent transparent;
        }

        .tooltip:hover .tooltiptext {
            visibility: visible;
            opacity: 1;
        }
    </style>
</head>
<body>
    <div class="dashboard">
        <div class="status-header">
            <div>
                <h1>Vinkeljernet Health Dashboard</h1>
                <p class="timestamp" id="timestamp">Last updated: -</p>
            </div>
            <div class="status-indicator">
                <div class="status-circle" id="status-circle"></div>
                <span id="status-text">Loading...</span>
            </div>
        </div>
        
        <div class="actions">
            <button id="refresh-btn" onclick="refreshData()">
                <span id="loading-indicator" style="display: none;" class="loading"></span>
                Refresh
            </button>
            <button id="reset-services-btn" onclick="resetServices()">Reset Degraded Services</button>
            <div style="margin-left: auto; display: flex; align-items: center;">
                <span style="margin-right: 10px;">Auto refresh:</span>
                <label class="switch">
                    <input type="checkbox" id="auto-refresh-toggle" checked>
                    <span class="slider"></span>
                </label>
                <select id="refresh-interval" style="margin-left: 10px;">
                    <option value="5000">5s</option>
                    <option value="10000" selected>10s</option>
                    <option value="30000">30s</option>
                    <option value="60000">1m</option>
                </select>
            </div>
        </div>
        
        <h2>Services</h2>
        <div class="services-grid" id="services-container">
            <div class="service-card">
                <div class="service-header">
                    <div class="service-name">Loading...</div>
                    <div class="service-status"></div>
                </div>
            </div>
        </div>
        
        <h2>Active Issues</h2>
        <div class="issues-container" id="issues-container">
            <p>Loading issues...</p>
        </div>
        
        <div class="history-container">
            <h2>Status History</h2>
            <div id="history-container">
                <p>Loading history...</p>
            </div>
        </div>
    </div>

    <script>
        let autoRefreshInterval;
        let currentStatus = {};
        
        // Initial load
        document.addEventListener('DOMContentLoaded', function() {
            refreshData();
            setupAutoRefresh();
            
            // Set up event listeners
            document.getElementById('auto-refresh-toggle').addEventListener('change', setupAutoRefresh);
            document.getElementById('refresh-interval').addEventListener('change', setupAutoRefresh);
        });
        
        function setupAutoRefresh() {
            // Clear existing interval if any
            if (autoRefreshInterval) {
                clearInterval(autoRefreshInterval);
            }
            
            // Set up new interval if toggle is checked
            const autoRefreshEnabled = document.getElementById('auto-refresh-toggle').checked;
            if (autoRefreshEnabled) {
                const interval = parseInt(document.getElementById('refresh-interval').value);
                autoRefreshInterval = setInterval(refreshData, interval);
            }
        }
        
        async function refreshData() {
            try {
                // Show loading indicator
                document.getElementById('loading-indicator').style.display = 'inline-block';
                document.getElementById('refresh-btn').disabled = true;
                
                // Fetch the latest data
                const response = await fetch('/api/status');
                const data = await response.json();
                currentStatus = data;
                
                // Update UI
                updateStatusHeader(data);
                updateServicesGrid(data);
                updateIssues(data);
                updateHistory();
                
                console.log('Dashboard refreshed with new data');
            } catch (error) {
                console.error('Failed to refresh dashboard:', error);
            } finally {
                // Hide loading indicator
                document.getElementById('loading-indicator').style.display = 'none';
                document.getElementById('refresh-btn').disabled = false;
            }
        }
        
        function updateStatusHeader(data) {
            const timestamp = new Date(data.timestamp).toLocaleString();
            document.getElementById('timestamp').textContent = `Last updated: ${timestamp}`;
            
            const statusCircle = document.getElementById('status-circle');
            const statusText = document.getElementById('status-text');
            
            // Remove all classes and add the appropriate one
            statusCircle.className = 'status-circle';
            statusCircle.classList.add(data.overall_health);
            
            // Update status text with appropriate formatting based on health
            let statusMessage = data.overall_health.charAt(0).toUpperCase() + data.overall_health.slice(1);
            if (data.overall_health !== 'healthy') {
                statusMessage += ` (${data.issues.length} issue${data.issues.length !== 1 ? 's' : ''})`;
            }
            statusText.textContent = statusMessage;
        }
        
        function updateServicesGrid(data) {
            const servicesContainer = document.getElementById('services-container');
            servicesContainer.innerHTML = ''; // Clear existing content
            
            // Sort services by health status (unhealthy first)
            const services = Object.entries(data.services).sort((a, b) => {
                const aHealthy = a[1].health;
                const bHealthy = b[1].health;
                if (aHealthy === bHealthy) return a[0].localeCompare(b[0]);
                return aHealthy ? 1 : -1; // Unhealthy first
            });
            
            services.forEach(([name, service]) => {
                const status = service.health ? 'healthy' : 'degraded';
                const circuitState = service.circuit_state || 'unknown';
                
                // Create service card
                const serviceCard = document.createElement('div');
                serviceCard.className = 'service-card';
                
                // Service header with name and status
                const headerDiv = document.createElement('div');
                headerDiv.className = 'service-header';
                
                const nameDiv = document.createElement('div');
                nameDiv.className = 'service-name';
                nameDiv.textContent = name;
                
                const statusDiv = document.createElement('div');
                statusDiv.className = `service-status ${status}`;
                statusDiv.textContent = status;
                
                headerDiv.appendChild(nameDiv);
                headerDiv.appendChild(statusDiv);
                serviceCard.appendChild(headerDiv);
                
                // Circuit state
                const circuitDiv = document.createElement('div');
                circuitDiv.className = 'circuit-state';
                circuitDiv.innerHTML = `Circuit: <span class="${circuitState}">${circuitState}</span>`;
                serviceCard.appendChild(circuitDiv);
                
                // Last error if available
                if (service.last_error) {
                    const errorRow = document.createElement('div');
                    errorRow.className = 'detail-row tooltip';
                    
                    const errorLabel = document.createElement('div');
                    errorLabel.className = 'detail-label';
                    errorLabel.textContent = 'Last error:';
                    
                    const errorTime = document.createElement('div');
                    let displayTime = 'Unknown';
                    if (service.error_time) {
                        try {
                            const errorDate = new Date(service.error_time);
                            displayTime = errorDate.toLocaleTimeString();
                        } catch (e) {
                            displayTime = service.error_time;
                        }
                    }
                    errorTime.textContent = displayTime;
                    
                    // Tooltip with full error message
                    const tooltip = document.createElement('span');
                    tooltip.className = 'tooltiptext';
                    tooltip.textContent = service.last_error;
                    
                    errorRow.appendChild(errorLabel);
                    errorRow.appendChild(errorTime);
                    errorRow.appendChild(tooltip);
                    serviceCard.appendChild(errorRow);
                }
                
                // Cache stats if available
                if (service.cache) {
                    const cacheRow = document.createElement('div');
                    cacheRow.className = 'detail-row';
                    
                    const cacheLabel = document.createElement('div');
                    cacheLabel.className = 'detail-label';
                    cacheLabel.textContent = 'Cache hit rate:';
                    
                    const cacheValue = document.createElement('div');
                    cacheValue.textContent = service.cache.hit_rate;
                    if (service.cache.degraded_mode) {
                        cacheValue.innerHTML += ' <span style="color:#e67e22">(degraded)</span>';
                    }
                    
                    cacheRow.appendChild(cacheLabel);
                    cacheRow.appendChild(cacheValue);
                    serviceCard.appendChild(cacheRow);
                }
                
                // Reset button for degraded services
                if (!service.health) {
                    const resetRow = document.createElement('div');
                    resetRow.style.marginTop = '12px';
                    resetRow.style.textAlign = 'right';
                    
                    const resetBtn = document.createElement('button');
                    resetBtn.textContent = 'Reset Service';
                    resetBtn.style.fontSize = '0.8em';
                    resetBtn.style.padding = '4px 8px';
                    resetBtn.onclick = function() { resetService(name); };
                    
                    resetRow.appendChild(resetBtn);
                    serviceCard.appendChild(resetRow);
                }
                
                servicesContainer.appendChild(serviceCard);
            });
        }
        
        function updateIssues(data) {
            const issuesContainer = document.getElementById('issues-container');
            
            if (!data.issues || data.issues.length === 0) {
                issuesContainer.innerHTML = '<p>No active issues 👍</p>';
                return;
            }
            
            issuesContainer.innerHTML = ''; // Clear existing content
            
            data.issues.forEach(issue => {
                const issueDiv = document.createElement('div');
                issueDiv.className = `issue ${issue.severity}`;
                
                const issueText = document.createElement('div');
                issueText.innerHTML = `<strong>${issue.component}:</strong> ${issue.message}`;
                
                issueDiv.appendChild(issueText);
                issuesContainer.appendChild(issueDiv);
            });
        }
        
        async function updateHistory() {
            try {
                const response = await fetch('/api/history');
                const historyData = await response.json();
                
                const historyContainer = document.getElementById('history-container');
                historyContainer.innerHTML = ''; // Clear existing content
                
                if (historyData.length === 0) {
                    historyContainer.innerHTML = '<p>No history data available</p>';
                    return;
                }
                
                historyData.slice().reverse().forEach(entry => {
                    const timestamp = new Date(entry.timestamp).toLocaleString();
                    const itemDiv = document.createElement('div');
                    itemDiv.className = 'history-item';
                    
                    const statusCircle = document.createElement('div');
                    statusCircle.className = `status-circle ${entry.overall_health}`;
                    statusCircle.style.width = '12px';
                    statusCircle.style.height = '12px';
                    
                    const statusText = document.createElement('div');
                    statusText.style.marginLeft = '10px';
                    statusText.textContent = `${timestamp}: ${entry.overall_health.toUpperCase()}`;
                    
                    if (entry.issues && entry.issues.length > 0) {
                        statusText.textContent += ` (${entry.issues.length} issues)`;
                    }
                    
                    itemDiv.appendChild(statusCircle);
                    itemDiv.appendChild(statusText);
                    historyContainer.appendChild(itemDiv);
                });
            } catch (error) {
                console.error('Failed to fetch history:', error);
            }
        }
        
        async function resetServices() {
            try {
                const response = await fetch('/api/reset-degraded', {
                    method: 'POST'
                });
                
                if (response.ok) {
                    // Refresh after a short delay to allow services to reset
                    setTimeout(refreshData, 1000);
                } else {
                    console.error('Failed to reset services');
                }
            } catch (error) {
                console.error('Error resetting services:', error);
            }
        }
        
        async function resetService(name) {
            try {
                const response = await fetch(`/api/reset-service/${name}`, {
                    method: 'POST'
                });
                
                if (response.ok) {
                    // Refresh after a short delay
                    setTimeout(refreshData, 500);
                } else {
                    console.error(`Failed to reset service ${name}`);
                }
            } catch (error) {
                console.error(`Error resetting service ${name}:`, error);
            }
        }
    </script>
</body>
</html>
"""


class HealthDashboard:
    """
    A web-based health dashboard for monitoring Vinkeljernet services.
    
    This dashboard provides real-time visualization of service health,
    circuit breaker status, and system issues. It allows operators to
    monitor the system and take corrective actions when needed.
    """
    
    def __init__(self):
        """Initialize the health dashboard."""
        self.app = None
        self.app_status = get_app_status()
        self.server_thread = None
        self.is_running = False
        self.port = 8080
        self.template = DEFAULT_TEMPLATE
        self.template_dir = os.path.join(os.path.dirname(__file__), 'templates')
    
    def _create_app(self) -> Flask:
        """Create and configure the Flask application"""
        if not FLASK_AVAILABLE:
            raise ImportError("Flask is required for the health dashboard. "
                             "Install it with: pip install flask")
        
        app = Flask(__name__)
        
        # Routes
        @app.route('/')
        def index():
            return self.template
        
        @app.route('/api/status')
        def get_status():
            return jsonify(self.app_status.get_dashboard_data())
        
        @app.route('/api/history')
        def get_history():
            return jsonify(self.app_status.get_status_history())
        
        @app.route('/api/reset-degraded', methods=['POST'])
        def reset_degraded():
            degraded_services = self.app_status.get_degraded_services()
            result = {"services_reset": []}
            
            for service_name in degraded_services:
                success = self.app_status.reset_service(service_name)
                if success:
                    result["services_reset"].append(service_name)
            
            return jsonify(result)
        
        @app.route('/api/reset-service/<service_name>', methods=['POST'])
        def reset_service(service_name):
            success = self.app_status.reset_service(service_name)
            return jsonify({"success": success})
        
        @app.route('/api/detailed-status')
        def detailed_status():
            return jsonify({
                "app_status": self.app_status.get_current_status(),
                "services": get_all_services_status()
            })
        
        return app
    
    def start(self, port: int = 8080, open_browser: bool = True) -> None:
        """
        Start the health dashboard server.
        
        Args:
            port: Port to run the server on
            open_browser: Whether to automatically open the dashboard in a browser
        """
        if self.is_running:
            logger.warning("Health dashboard is already running")
            return
        
        if not FLASK_AVAILABLE:
            logger.error("Flask is required for the health dashboard. "
                        "Install it with: pip install flask")
            return
        
        try:
            self.port = port
            self.app = self._create_app()
            
            # Create and start server thread
            def run_server():
                try:
                    self.is_running = True
                    self.app.run(host='0.0.0.0', port=port, debug=False, use_reloader=False)
                except Exception as e:
                    logger.error(f"Error in health dashboard server: {e}")
                finally:
                    self.is_running = False
            
            self.server_thread = threading.Thread(target=run_server)
            self.server_thread.daemon = True
            self.server_thread.start()
            
            # Open browser if requested
            if open_browser:
                webbrowser.open(f"http://localhost:{port}")
            
            logger.info(f"Health dashboard started on http://localhost:{port}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to start health dashboard: {e}")
            return False
    
    def stop(self) -> None:
        """Stop the health dashboard server"""
        if not self.is_running:
            logger.warning("Health dashboard is not running")
            return
        
        self.is_running = False
        
        # Flask doesn't have a clean shutdown in this usage, so we'll just
        # let the daemon thread terminate when the program exits
        logger.info("Health dashboard stopping")
    
    def set_template(self, template_html: str) -> None:
        """Set a custom HTML template for the dashboard"""
        self.template = template_html
    
    def load_template_from_file(self, file_path: str) -> bool:
        """Load dashboard template from a file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                self.template = f.read()
            return True
        except Exception as e:
            logger.error(f"Failed to load template: {e}")
            return False


# Singleton instance
_dashboard_instance = None


def get_dashboard() -> HealthDashboard:
    """
    Get the global dashboard instance.
    
    Returns:
        HealthDashboard: The global dashboard instance
    """
    global _dashboard_instance
    
    if _dashboard_instance is None:
        _dashboard_instance = HealthDashboard()
        
    return _dashboard_instance


def start_dashboard(port: int = 8080, open_browser: bool = True) -> None:
    """
    Start the health dashboard.
    
    Args:
        port: Port to run the dashboard on
        open_browser: Whether to open the dashboard in a browser
    """
    dashboard = get_dashboard()
    return dashboard.start(port, open_browser)


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Start the dashboard
    dashboard = HealthDashboard()
    dashboard.start(port=8080)
    
    print("Health dashboard running on http://localhost:8080")
    print("Press Ctrl+C to exit")
    
    try:
        # Keep the main thread alive
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Shutting down...")