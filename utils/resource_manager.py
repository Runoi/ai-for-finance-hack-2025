import time
import psutil
import os
from typing import Dict, Any

class ResourceManager:
    """
    Центральный трекер для мониторинга времени, API-затрат и системных ресурсов.
    Работает как синглтон, создается один раз при старте.
    """
    def __init__(self, api_budget_usd: float = 3.0, time_limit_sec: int = 4 * 3600):
        self.api_budget_usd = api_budget_usd
        self.time_limit_sec = time_limit_sec
        
        self.start_time = time.time()
        self.api_cost_usd = 0.0
        
        self.process = psutil.Process(os.getpid())
        self.initial_ram_usage_mb = self.process.memory_info().rss / (1024 * 1024)
        self.peak_ram_usage_mb = self.initial_ram_usage_mb
        
        self.logs = [f"[SYS] Initial RAM usage: {self.initial_ram_usage_mb:.2f} MB"]
        print(self.logs[-1])

    def _update_peak_ram(self) -> float:
        """Внутренний метод для обновления пикового потребления RAM."""
        current_ram_mb = self.process.memory_info().rss / (1024 * 1024)
        if current_ram_mb > self.peak_ram_usage_mb:
            self.peak_ram_usage_mb = current_ram_mb
        return current_ram_mb

    def log_checkpoint(self, context_message: str):
        """
        Универсальный метод для логирования состояния системы в контрольной точке.
        Замеряет время, CPU и RAM.
        """
        elapsed = time.time() - self.start_time
        current_ram_mb = self._update_peak_ram()
        # interval > 0 необходим для неблокирующего замера CPU
        cpu_percent = self.process.cpu_percent(interval=0.1) 

        log_entry = (
            f"[{elapsed:8.2f}s] CHECKPOINT: {context_message:<40} | "
            f"CPU: {cpu_percent:5.1f}% | "
            f"RAM: {current_ram_mb:8.2f} MB | "
            f"Peak RAM: {self.peak_ram_usage_mb:8.2f} MB"
        )
        print(log_entry)
        self.logs.append(log_entry)

        if elapsed > self.time_limit_sec:
            print("!!! CRITICAL WARNING: TIME LIMIT EXCEEDED !!!")

    def log_api_call(self, model_name: str, cost_usd: float):
        """Логирует вызов API и обновляет общий счетчик затрат."""
        self.api_cost_usd += cost_usd
        
        log_entry = (
            f"[{time.time() - self.start_time:8.2f}s] [API CALL] Model: {model_name:<45} | "
            f"Cost: ${cost_usd:.5f} | "
            f"Total Spent: ${self.api_cost_usd:.5f} / ${self.api_budget_usd}"
        )
        print(log_entry)
        self.logs.append(log_entry)
        
        if self.api_cost_usd > self.api_budget_usd:
            print("!!! CRITICAL WARNING: API BUDGET EXCEEDED !!!")

    def get_summary(self) -> Dict[str, Any]:
        """Возвращает финальную сводку по использованным ресурсам."""
        final_ram_mb = self.process.memory_info().rss / (1024 * 1024)
        return {
            "api_spent_usd": self.api_cost_usd,
            "time_elapsed_sec": time.time() - self.start_time,
            "peak_ram_usage_mb": self.peak_ram_usage_mb,
            "ram_increase_mb": final_ram_mb - self.initial_ram_usage_mb
        }

# Глобальный экземпляр, который будет импортироваться в других модулях.
resource_manager = ResourceManager()