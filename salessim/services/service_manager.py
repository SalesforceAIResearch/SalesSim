#!/usr/bin/env python3

import asyncio
import subprocess
import time
import logging
import aiohttp
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

class ServiceManager:
    """Manages the lifecycle of microservices"""

    def __init__(self):
        self.services: Dict[str, subprocess.Popen] = {}
        self.service_configs = {
            "lookup_service": {
                "script": "salessim/services/sales_service.py",
                "port": 8001,
                "health_endpoint": "http://127.0.0.1:8001/health"
            }
        }

    async def start_service(self, service_name: str) -> bool:
        """Start a specific service"""
        if service_name in self.services:
            logger.warning(f"Service {service_name} is already running")
            return True

        if service_name not in self.service_configs:
            logger.error(f"Unknown service: {service_name}")
            return False

        config = self.service_configs[service_name]
        try:
            logger.info(f"Starting {service_name} service...")
            stdout_file = open(f"{service_name}_stdout.log", "a")
            stderr_file = open(f"{service_name}_stderr.log", "a")
            process = subprocess.Popen([
                "python3", "-m", config["script"].replace("/", ".").replace(".py", "")
            ], stdout=stdout_file, stderr=stderr_file)

            self.services[service_name] = process

            # Wait for service to be ready
            if await self._wait_for_service_health(service_name, timeout=30):
                logger.info(f"Service {service_name} started successfully")
                return True
            else:
                logger.error(f"Service {service_name} failed to start")
                await self.stop_service(service_name)
                return False

        except Exception as e:
            logger.error(f"Failed to start service {service_name}: {e}")
            return False

    async def stop_service(self, service_name: str) -> bool:
        """Stop a specific service"""
        if service_name not in self.services:
            logger.warning(f"Service {service_name} is not running")
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
            logger.info(f"Service {service_name} stopped")
            return True

        except Exception as e:
            logger.error(f"Failed to stop service {service_name}: {e}")
            return False

    async def start_all_services(self) -> bool:
        """Start all configured services"""
        logger.info("Starting all services...")
        success = True

        for service_name in self.service_configs:
            if not await self.start_service(service_name):
                success = False

        return success

    async def stop_all_services(self) -> bool:
        """Stop all running services"""
        logger.info("Stopping all services...")
        success = True

        for service_name in list(self.services.keys()):
            if not await self.stop_service(service_name):
                success = False

        return success

    async def _wait_for_service_health(self, service_name: str, timeout: int = 30) -> bool:
        """Wait for a service to respond to health checks"""
        config = self.service_configs[service_name]
        health_url = config["health_endpoint"]

        start_time = time.time()
        session = aiohttp.ClientSession()
        try:
            while time.time() - start_time < timeout:
                try:
                    async with session.get(health_url, timeout=aiohttp.ClientTimeout(total=2)) as response:
                        if response.status == 200:
                            return True
                except:
                    pass

                await asyncio.sleep(1)

            return False
        finally:
            await session.close()

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