"""
Rate limiter for API calls that respects both RPM (requests per minute) and TPM (tokens per minute) limits.
"""
import threading
import time
from collections import deque
from typing import Optional


class RateLimiter:
    """
    Thread-safe rate limiter that tracks both requests per minute (RPM) and tokens per minute (TPM).
    Uses a sliding window approach to accurately track usage.
    """

    def __init__(
        self,
        rpm_limit: Optional[int] = None,
        tpm_limit: Optional[int] = None,
        window_seconds: float = 60.0,
    ):
        """
        Initialize rate limiter.

        Args:
            rpm_limit: Maximum requests per minute (None = unlimited)
            tpm_limit: Maximum tokens per minute (None = unlimited)
            window_seconds: Time window for rate limiting (default: 60 seconds)
        """
        self.rpm_limit = rpm_limit
        self.tpm_limit = tpm_limit
        self.window_seconds = window_seconds

        # Track request timestamps
        self.request_times = deque()

        # Track token usage: (timestamp, token_count) pairs
        self.token_usage = deque()

        # Thread lock for thread safety
        self.lock = threading.Lock()

    def _clean_old_entries(self, current_time: float):
        """Remove entries older than the time window."""
        cutoff_time = current_time - self.window_seconds

        # Clean old requests
        while self.request_times and self.request_times[0] < cutoff_time:
            self.request_times.popleft()

        # Clean old token usage
        while self.token_usage and self.token_usage[0][0] < cutoff_time:
            self.token_usage.popleft()

    def _get_current_usage(self, current_time: float) -> tuple[int, int]:
        """Get current RPM and TPM usage within the time window."""
        self._clean_old_entries(current_time)

        current_rpm = len(self.request_times)
        current_tpm = sum(tokens for _, tokens in self.token_usage)

        return current_rpm, current_tpm

    def _calculate_wait_time(self, estimated_tokens: int) -> float:
        """
        Calculate how long to wait before making a request.

        Args:
            estimated_tokens: Estimated tokens for the upcoming request

        Returns:
            Number of seconds to wait (0 if no wait needed)
        """
        current_time = time.time()
        current_rpm, current_tpm = self._get_current_usage(current_time)

        wait_time = 0.0

        # Check RPM limit
        if self.rpm_limit and current_rpm >= self.rpm_limit:
            # Calculate when the oldest request will expire
            oldest_request_time = self.request_times[0]
            wait_for_rpm = oldest_request_time + self.window_seconds - current_time
            wait_time = max(wait_time, wait_for_rpm)

        # Check TPM limit
        if self.tpm_limit and (current_tpm + estimated_tokens) > self.tpm_limit:
            # Calculate when enough tokens will be freed
            tokens_needed = (current_tpm + estimated_tokens) - self.tpm_limit
            tokens_freed = 0

            for timestamp, tokens in self.token_usage:
                tokens_freed += tokens
                if tokens_freed >= tokens_needed:
                    wait_for_tpm = timestamp + self.window_seconds - current_time
                    wait_time = max(wait_time, wait_for_tpm)
                    break

        # Add a small buffer to avoid edge cases
        if wait_time > 0:
            wait_time += 0.1

        return wait_time

    def acquire(self):
        """
        Check if we can make an API request. Will sleep if necessary to respect rate limits.
        Call this BEFORE making an API request.
        """
        with self.lock:
            # Check only RPM limit (we don't know tokens yet)
            current_time = time.time()
            self._clean_old_entries(current_time)

            current_rpm = len(self.request_times)

            wait_time = 0.0

            # Check RPM limit
            if self.rpm_limit and current_rpm >= self.rpm_limit:
                oldest_request_time = self.request_times[0]
                wait_time = oldest_request_time + self.window_seconds - current_time + 0.1

            if wait_time > 0:
                print(f"Rate limiter: waiting {wait_time:.2f}s to respect RPM limit...")
                time.sleep(wait_time)

            # Record this request
            current_time = time.time()
            self.request_times.append(current_time)

    def report_actual_usage(self, actual_tokens: int):
        """
        Report actual token usage AFTER receiving API response.
        Checks if we need to wait before next request to respect TPM limits.

        Args:
            actual_tokens: Actual number of tokens used (from API response)
        """
        with self.lock:
            current_time = time.time()
            self._clean_old_entries(current_time)

            # Record the actual token usage
            self.token_usage.append((current_time, actual_tokens))

            # Check if we're over TPM limit
            if self.tpm_limit:
                current_tpm = sum(tokens for _, tokens in self.token_usage)

                if current_tpm > self.tpm_limit:
                    # Calculate how long to wait for tokens to free up
                    tokens_to_free = current_tpm - self.tpm_limit
                    tokens_freed = 0

                    for timestamp, tokens in self.token_usage:
                        tokens_freed += tokens
                        if tokens_freed >= tokens_to_free:
                            wait_time = timestamp + self.window_seconds - current_time + 0.1
                            if wait_time > 0:
                                print(f"Rate limiter: TPM limit exceeded ({current_tpm}/{self.tpm_limit}), waiting {wait_time:.2f}s...")
                                time.sleep(wait_time)
                            break

    def get_current_stats(self) -> dict:
        """Get current rate limiter statistics."""
        with self.lock:
            current_time = time.time()
            current_rpm, current_tpm = self._get_current_usage(current_time)

            return {
                "current_rpm": current_rpm,
                "rpm_limit": self.rpm_limit,
                "current_tpm": current_tpm,
                "tpm_limit": self.tpm_limit,
                "rpm_utilization": (current_rpm / self.rpm_limit * 100) if self.rpm_limit else 0,
                "tpm_utilization": (current_tpm / self.tpm_limit * 100) if self.tpm_limit else 0,
            }
