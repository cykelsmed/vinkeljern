"""
System health monitoring for Vinkeljernet.

This module extends the fault tolerance architecture with application-wide
status monitoring, health checks, and automatic recovery mechanisms.
"""

import os
import time
import asyncio
import logging
import threading
from typing import Dict, List, Any, Callable, Optional, Set, Union
from datetime import datetime, timedelta
from enum import Enum

from fault_tolerance import (
    CircuitBreaker,
    FaultTolerantService,
    get_all_services_status,
    get_service
)

# Configure logging
logger = logging.getLogger("vinkeljernet.app_status")

# Global app status instance
_app_status_instance = None


class HealthStatus(Enum):
    """Health status levels"""
    HEALTHY = "healthy"
    WARNING = "warning"
    DEGRADED = "degraded"
    ERROR = "error"


class VinkeljernetAppStatus:
    """
    Global application status monitor for Vinkeljernet.
    
    This class tracks the health of all system components and provides
    a central place to check the overall system status, run health checks,
    and implement automatic recovery actions.
    """
    
    def __init__(self):
        """Initialize the application status monitor."""
        # Components registry
        self._components = {}
        self._health_checks = []
        self._recovery_actions = []
        self._status_history = []
        
        # State variables
        self._degraded_mode = False
        self._degraded_services = set()
        self._last_health_check = None
        self._health_check_running = False
        self._status_lock = threading.RLock()
        self._status_history_max_length = 100
        
        logger.info("Initialized VinkeljernetAppStatus")
    
    def register_component(self, name: str, component: Any) -> None:
        """
        Register a component with the application status.
        
        Args:
            name: Name of the component
            component: Component object (service, circuit breaker, etc.)
        """
        self._components[name] = component
        logger.info(f"Registered component: {name}")
    
    def register_health_check(self, check_function: Callable[[], Dict[str, Any]]) -> None:
        """
        Register a health check function.
        
        Args:
            check_function: Function that performs a health check and returns status
        """
        self._health_checks.append(check_function)
    
    def register_recovery_action(
        self, 
        condition_function: Callable[[], bool],
        action_function: Callable[[], None],
        frequency_seconds: int = 300
    ) -> None:
        """
        Register an automatic recovery action.
        
        Args:
            condition_function: Function that determines if action should run
            action_function: Function that performs recovery action
            frequency_seconds: Minimum time between action executions
        """
        self._recovery_actions.append({
            "condition": condition_function,
            "action": action_function,
            "frequency_seconds": frequency_seconds,
            "last_run": None
        })
    
    def run_health_check(self) -> Dict[str, Any]:
        """
        Run all registered health checks and update status.
        
        Returns:
            Dict with health check results
        """
        with self._status_lock:
            if self._health_check_running:
                logger.debug("Health check already running, skipping")
                if self._last_health_check:
                    return self._last_health_check
                return {"overall_health": "unknown", "message": "Health check in progress"}
            
            self._health_check_running = True
        
        try:
            now = datetime.now()
            status = {
                "timestamp": now.isoformat(),
                "overall_health": "healthy",
                "services": {},
                "issues": [],
                "degraded_services": list(self._degraded_services)
            }
            
            # Check registered services
            for name, component in self._components.items():
                try:
                    if hasattr(component, "get_status") and callable(component.get_status):
                        service_status = component.get_status()
                    elif hasattr(component, "get_stats") and callable(component.get_stats):
                        service_status = component.get_stats()
                    else:
                        service_status = {"status": "unknown", "component_type": type(component).__name__}
                    
                    status["services"][name] = service_status
                    
                    # Check for issues
                    is_healthy = service_status.get("healthy", True)
                    circuit_state = service_status.get("circuit", {}).get("state", "closed") if isinstance(service_status.get("circuit"), dict) else service_status.get("circuit", "closed")
                    
                    if not is_healthy or circuit_state == "open":
                        status["issues"].append({
                            "component": name,
                            "severity": "error" if not is_healthy else "warning",
                            "message": f"Service {name} is unhealthy" if not is_healthy else f"Circuit {name} is open"
                        })
                        
                        if name not in self._degraded_services and not is_healthy:
                            self._degraded_services.add(name)
                    elif name in self._degraded_services and is_healthy:
                        self._degraded_services.discard(name)
                        
                except Exception as e:
                    logger.error(f"Error getting status for component {name}: {e}")
                    status["issues"].append({
                        "component": name,
                        "severity": "error",
                        "message": f"Error getting status: {str(e)}"
                    })
            
            # Run registered health checks
            for i, check_func in enumerate(self._health_checks):
                try:
                    result = check_func()
                    name = result.get("service", f"health_check_{i}")
                    status["services"][name] = result
                    
                    if not result.get("healthy", True):
                        status["issues"].append({
                            "component": name,
                            "severity": result.get("severity", "error"),
                            "message": result.get("message", f"Health check for {name} failed")
                        })
                except Exception as e:
                    logger.error(f"Error running health check {i}: {e}")
                    status["issues"].append({
                        "component": f"health_check_{i}",
                        "severity": "error",
                        "message": f"Error running health check: {str(e)}"
                    })
                    
            # Evaluate overall status based on issues
            if status["issues"]:
                error_count = sum(1 for issue in status["issues"] if issue.get("severity") == "error")
                warning_count = sum(1 for issue in status["issues"] if issue.get("severity") == "warning")
                
                if error_count > 0:
                    if error_count >= 3:  # Multiple critical failures
                        status["overall_health"] = "error"
                    else:
                        status["overall_health"] = "degraded"
                elif warning_count > 0:
                    if warning_count >= 3:  # Multiple warnings
                        status["overall_health"] = "degraded"
                    else:
                        status["overall_health"] = "warning"
            
            # Update degraded mode flag
            self._degraded_mode = (status["overall_health"] in ["degraded", "error"])
            
            # Save status to history
            self._last_health_check = status
            self._status_history.append(status)
            
            # Trim history if needed
            if len(self._status_history) > self._status_history_max_length:
                self._status_history = self._status_history[-self._status_history_max_length:]
            
            return status
            
        finally:
            self._health_check_running = False
    
    def is_degraded(self) -> bool:
        """
        Check if the system is in degraded mode.
        
        Returns:
            bool: True if system is degraded
        """
        return self._degraded_mode
    
    def get_degraded_services(self) -> Set[str]:
        """Get names of services that are currently degraded"""
        return self._degraded_services
    
    def get_status_history(self) -> List[Dict[str, Any]]:
        """Get the history of status checks"""
        return self._status_history
    
    def get_current_status(self) -> Dict[str, Any]:
        """
        Get the current system status.
        
        Returns:
            Dict with current status
        """
        if self._last_health_check is None:
            return self.run_health_check()
        return self._last_health_check
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """
        Get data for status dashboard display.
        
        Returns:
            Dict with dashboard data
        """
        current_status = self.get_current_status()
        
        # Get service-specific dashboard data
        services_data = {}
        for name, service_info in current_status.get("services", {}).items():
            # Simplify data for dashboard
            services_data[name] = {
                "health": service_info.get("healthy", True),
                "status": "healthy" if service_info.get("healthy", True) else "degraded",
                "circuit_state": service_info.get("circuit", "closed") if isinstance(service_info.get("circuit"), str) else service_info.get("circuit", {}).get("state", "closed"),
                "last_error": service_info.get("last_error_message", None),
                "error_time": service_info.get("last_error", None)
            }
            
            # Add cache stats if available
            if "cache_stats" in service_info:
                cache_stats = service_info["cache_stats"]
                services_data[name]["cache"] = {
                    "hit_rate": cache_stats.get("hit_rate", "0%"),
                    "degraded_mode": cache_stats.get("degraded_mode", False)
                }
        
        return {
            "timestamp": datetime.now().isoformat(),
            "overall_health": current_status.get("overall_health", "unknown"),
            "services": services_data,
            "issue_count": len(current_status.get("issues", [])),
            "degraded_services": list(self._degraded_services)
        }
    
    async def run_recovery_actions(self) -> List[str]:
        """
        Run registered recovery actions if their conditions are met.
        
        Returns:
            List of actions that were executed
        """
        executed_actions = []
        now = datetime.now()
        
        for action_info in self._recovery_actions:
            try:
                # Check if it's time to run this action
                last_run = action_info["last_run"]
                freq = action_info["frequency_seconds"]
                
                if last_run is not None and now - last_run < timedelta(seconds=freq):
                    continue  # Too soon to run again
                
                # Check if condition is met
                if action_info["condition"]():
                    # Execute recovery action
                    action_info["action"]()
                    action_info["last_run"] = now
                    executed_actions.append(action_info["action"].__name__)
                    logger.info(f"Executed recovery action: {action_info['action'].__name__}")
            except Exception as e:
                logger.error(f"Error running recovery action: {str(e)}")
        
        return executed_actions
    
    def reset_service(self, name: str) -> bool:
        """
        Reset a specific service.
        
        Args:
            name: Service name
            
        Returns:
            bool: True if reset was successful
        """
        try:
            service = get_service(name)
            if hasattr(service, "reset") and callable(service.reset):
                service.reset()
                if name in self._degraded_services:
                    self._degraded_services.discard(name)
                logger.info(f"Reset service: {name}")
                return True
            return False
        except Exception as e:
            logger.error(f"Error resetting service {name}: {str(e)}")
            return False


def get_app_status() -> VinkeljernetAppStatus:
    """
    Get the global application status instance.
    
    Returns:
        VinkeljernetAppStatus: The global app status instance
    """
    global _app_status_instance
    
    if _app_status_instance is None:
        _app_status_instance = VinkeljernetAppStatus()
        
    return _app_status_instance


async def start_monitoring_loop(check_interval: int = 60):
    """
    Start a monitoring loop that periodically checks system health.
    
    Args:
        check_interval: Time between health checks in seconds
    """
    app_status = get_app_status()
    
    while True:
        try:
            app_status.run_health_check()
            await app_status.run_recovery_actions()
        except Exception as e:
            logger.error(f"Error in monitoring loop: {str(e)}")
            
        await asyncio.sleep(check_interval)