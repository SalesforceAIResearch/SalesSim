#!/usr/bin/env python3

import asyncio
import subprocess
import time
import logging
import os
import aiohttp
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class ServiceStartupError(RuntimeError):
    """Raised when a managed service fails during startup."""


class ServiceManager:
    """Manages the lifecycle of microservices"""

    def __init__(self):
        self.services: Dict[str, subprocess.Popen] = {}
        self.service_files: Dict[str, tuple] = {}  # Store file handles for cleanup
        self.service_configs = {
            "lookup_service": {
                "script": "salessim/services/sales_service.py",
                "port": os.environ.get("LOOKUP_BASE_PORT","8001"), 
                "health_endpoint": f"http://127.0.0.1:{os.environ.get('LOOKUP_BASE_PORT', '8001')}/health"
            }
        }

    async def start_service(self, service_name: str) -> bool:
        """Start a specific service"""
        if service_name in self.services:
            logger.warning(f"Service {service_name} is already running")
            return True

        if service_name not in self.service_configs:
            raise ServiceStartupError(f"Unknown service: {service_name}")
        config = self.service_configs[service_name]

        # First, check if service is already running (manual start)
        if await self._check_service_health_once(service_name):
            logger.info(f"Service {service_name} is already running (detected via health check)")
            return True

        try:
            logger.info(f"Starting {service_name} service...")
            stdout_file = open(self._stdout_log_path(service_name), "a")
            stderr_file = open(self._stderr_log_path(service_name), "a")
            env = os.environ.copy()
            if service_name == "lookup_service":
                lookup_cuda_visible = os.environ.get("LOOKUP_CUDA_VISIBLE_DEVICES")
                if lookup_cuda_visible is not None:
                    env["CUDA_VISIBLE_DEVICES"] = lookup_cuda_visible
            process = subprocess.Popen([
                "python3", "-m", config["script"].replace("/", ".").replace(".py", "")
            ], stdout=stdout_file, stderr=stderr_file, env=env)

            self.services[service_name] = process
            self.service_files[service_name] = (stdout_file, stderr_file)

            # Wait for service to be ready
            await self._wait_for_service_health(service_name, timeout=30)
            logger.info(f"Service {service_name} started successfully")
            return True

        except Exception as e:
            logger.exception(f"Failed to start service {service_name}: {e}")
            if service_name in self.services:
                await self.stop_service(service_name)
            else:
                # Close files if they were opened before the process was tracked.
                if 'stdout_file' in locals():
                    stdout_file.close()
                if 'stderr_file' in locals():
                    stderr_file.close()
            if isinstance(e, ServiceStartupError):
                raise
            raise ServiceStartupError(f"Failed to start service {service_name}") from e

    async def stop_service(self, service_name: str) -> bool:
        """Stop a specific service"""
        if service_name not in self.services:
            logger.info(f"Service {service_name} was not started by this manager (possibly manual start), skipping stop")
            return True

        try:
            process = self.services[service_name]
            process.terminate()

            # Give it a moment to terminate gracefully
            await asyncio.sleep(1)

            if process.poll() is None:
                # Force kill if it didn't terminate
                process.kill()

            del self.services[service_name]

            # Close the log files
            if service_name in self.service_files:
                stdout_file, stderr_file = self.service_files[service_name]
                stdout_file.close()
                stderr_file.close()
                del self.service_files[service_name]

            logger.info(f"Service {service_name} stopped")
            return True

        except Exception as e:
            logger.error(f"Failed to stop service {service_name}: {e}")
            return False

    async def start_all_services(self) -> bool:
        """Start all configured services"""
        logger.info("Starting all services...")

        for service_name in self.service_configs:
            await self.start_service(service_name)

        return True

    async def stop_all_services(self) -> bool:
        """Stop all running services"""
        logger.info("Stopping all services...")
        success = True

        for service_name in list(self.services.keys()):
            if not await self.stop_service(service_name):
                success = False

        return success

    async def _check_service_health_once(self, service_name: str) -> bool:
        """Check if service is healthy with a single attempt"""
        config = self.service_configs[service_name]
        health_url = config["health_endpoint"]
        if service_name == "lookup_service":
            lookup_base_url = os.environ.get("LOOKUP_BASE_URL")
            if lookup_base_url:
                health_url = f"{lookup_base_url.rstrip('/')}/health"

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(health_url, timeout=aiohttp.ClientTimeout(total=2)) as response:
                    return response.status == 200
        except Exception:
            return False

    async def _wait_for_service_health(self, service_name: str, timeout: int = 30) -> None:
        """Wait for a service to respond to health checks"""
        config = self.service_configs[service_name]
        health_url = config["health_endpoint"]
        if service_name == "lookup_service":
            lookup_base_url = os.environ.get("LOOKUP_BASE_URL")
            if lookup_base_url:
                health_url = f"{lookup_base_url.rstrip('/')}/health"

        # Immediate check first
        if await self._check_service_health_once(service_name):
            logger.info(f"Service {service_name} is immediately healthy")
            return

        # Wait for timeout period
        logger.info(f"Service {service_name} not ready immediately, waiting up to {timeout}s...")
        start_time = time.time()
        session = aiohttp.ClientSession()
        last_error: Optional[BaseException] = None
        last_status: Optional[int] = None
        last_body: Optional[str] = None
        try:
            while time.time() - start_time < timeout:
                process = self.services.get(service_name)
                if process is not None and process.poll() is not None:
                    stderr_tail = self._tail_log(self._stderr_log_path(service_name))
                    stderr_detail = f"\nRecent stderr:\n{stderr_tail}" if stderr_tail else ""
                    raise ServiceStartupError(
                        f"Service {service_name} exited during startup with code {process.returncode}. "
                        f"Check {self._stderr_log_path(service_name)} for details."
                        f"{stderr_detail}"
                    )

                try:
                    async with session.get(health_url, timeout=aiohttp.ClientTimeout(total=2)) as response:
                        if response.status == 200:
                            logger.info(f"Service {service_name} became healthy after {time.time() - start_time:.1f}s")
                            return
                        last_status = response.status
                        last_body = await response.text()
                except Exception as e:
                    last_error = e

                await asyncio.sleep(1)

            # Build helpful error message with instructions to start manually
            port = config.get("port", "8001")
            script = config.get("script", "")
            detail = (
                f"Service {service_name} did not become healthy within {timeout}s.\n"
                f"Health check URL: {health_url}\n\n"
                f"To start the service manually, run:\n"
                f"  python3 -m {script.replace('/', '.').replace('.py', '')}\n\n"
                f"Or increase the timeout by setting a longer value in service_manager.py"
            )
            if last_status is not None:
                detail += f"\n\nLast response status: {last_status}"
                if last_body:
                    detail += f", body: {last_body[:500]}"
            elif last_error is not None:
                detail += f"\n\nLast error: {last_error!r}"
            raise ServiceStartupError(detail)
        finally:
            await session.close()

    def _stdout_log_path(self, service_name: str) -> str:
        return f"{service_name}_stdout.log"

    def _stderr_log_path(self, service_name: str) -> str:
        return f"{service_name}_stderr.log"

    def _tail_log(self, log_path: str, max_bytes: int = 4000) -> str:
        try:
            if not os.path.exists(log_path):
                return ""
            with open(log_path, "rb") as f:
                f.seek(0, os.SEEK_END)
                size = f.tell()
                f.seek(max(size - max_bytes, 0), os.SEEK_SET)
                return f.read().decode(errors="replace").strip()
        except OSError as e:
            logger.debug(f"Could not read service log {log_path}: {e}")
            return ""

    def get_service_status(self) -> Dict[str, str]:
        """Get the status of all services"""
        status = {}
        for service_name in self.service_configs:
            if service_name in self.services:
                process = self.services[service_name]
                if process.poll() is None:
                    status[service_name] = "running"
                else:
                    status[service_name] = "stopped"
                    # Clean up dead processes
                    del self.services[service_name]
            else:
                status[service_name] = "not_started"

        return status
