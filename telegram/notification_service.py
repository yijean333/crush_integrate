"""
Telegram Notification Service
==============================

Service untuk mengirim notifikasi ke Telegram saat crash terdeteksi
Mendukung text message dan photo dengan caption

Usage:
    from telegram.notification_service import TelegramNotifier
    
    notifier = TelegramNotifier()
    notifier.send_crash_alert(frame_id=1234, screenshot_path="crash_frame.jpg")
"""

import requests
import time
from pathlib import Path
from typing import Optional, Dict
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class TelegramNotifier:
    """Service untuk kirim notifikasi ke Telegram"""
    
    def __init__(self, bot_token: Optional[str] = None, chat_id: Optional[str] = None):
        """
        Initialize Telegram Notifier
        
        Args:
            bot_token: Telegram Bot Token (default: dari telegram.config)
            chat_id: Telegram Chat ID (default: dari telegram.config)
        """
        try:
            from telegram.config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, NOTIFICATION_ENABLED, RATE_LIMIT_SECONDS
        except ImportError:
            logger.error("telegram/config.py not found! Please create telegram/config.py with Telegram credentials")
            raise
        
        self.bot_token = bot_token or TELEGRAM_BOT_TOKEN
        self.chat_id = chat_id or TELEGRAM_CHAT_ID
        self.enabled = NOTIFICATION_ENABLED
        self.rate_limit_seconds = RATE_LIMIT_SECONDS
        
        # Rate limiting: track last notification time
        self.last_notification_time = 0
        
        if not self.bot_token or not self.chat_id:
            logger.warning("Telegram credentials not configured. Notifications will be disabled.")
            self.enabled = False
        
        self.base_url = f"https://api.telegram.org/bot{self.bot_token}"
    
    def _check_rate_limit(self) -> bool:
        """Check apakah sudah cukup waktu sejak notifikasi terakhir"""
        current_time = time.time()
        time_since_last = current_time - self.last_notification_time
        
        if time_since_last < self.rate_limit_seconds:
            remaining = self.rate_limit_seconds - time_since_last
            logger.warning(f"Rate limit: {remaining:.1f}s remaining (limit: {self.rate_limit_seconds}s)")
            print(f"⚠️ Telegram Rate Limit: {remaining:.1f}s remaining (limit: {self.rate_limit_seconds}s) - notification skipped")
            return False
        
        return True
    
    def send_message(self, text: str, parse_mode: str = "HTML") -> bool:
        """
        Kirim text message ke Telegram
        
        Args:
            text: Message text
            parse_mode: HTML atau Markdown (default: HTML)
        
        Returns:
            True jika berhasil, False jika gagal
        """
        if not self.enabled:
            logger.debug("Notifications disabled")
            return False
        
        if not self._check_rate_limit():
            logger.debug("Rate limit: skipping notification")
            return False
        
        url = f"{self.base_url}/sendMessage"
        data = {
            "chat_id": self.chat_id,
            "text": text,
            "parse_mode": parse_mode
        }
        
        try:
            response = requests.post(url, json=data, timeout=10)
            if response.status_code == 200:
                result = response.json()
                if result.get('ok'):
                    self.last_notification_time = time.time()
                    logger.info("Telegram notification sent successfully")
                    return True
                else:
                    error_desc = result.get('description', 'Unknown error')
                    logger.error(f"Telegram API error: {error_desc}")
                    # Print error untuk debugging
                    print(f"❌ Telegram API Error: {error_desc}")
                    return False
            else:
                error_text = response.text
                logger.error(f"HTTP error {response.status_code}: {error_text}")
                print(f"❌ Telegram HTTP Error {response.status_code}: {error_text}")
                return False
        except requests.exceptions.RequestException as e:
            logger.error(f"Error sending Telegram message: {e}")
            print(f"❌ Telegram Connection Error: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error sending Telegram message: {e}")
            print(f"❌ Telegram Unexpected Error: {e}")
            return False
    
    def send_photo(self, photo_path: str, caption: str = "", parse_mode: str = "HTML") -> bool:
        """
        Kirim photo dengan caption ke Telegram
        
        Args:
            photo_path: Path ke file foto
            caption: Caption untuk foto
            parse_mode: HTML atau Markdown (default: HTML)
        
        Returns:
            True jika berhasil, False jika gagal
        """
        if not self.enabled:
            logger.debug("Notifications disabled")
            return False
        
        if not self._check_rate_limit():
            logger.debug("Rate limit: skipping notification")
            return False
        
        photo_file = Path(photo_path)
        if not photo_file.exists():
            logger.error(f"Photo file not found: {photo_path}")
            return False
        
        url = f"{self.base_url}/sendPhoto"
        
        try:
            with open(photo_file, 'rb') as photo:
                files = {"photo": photo}
                data = {
                    "chat_id": self.chat_id,
                    "caption": caption,
                    "parse_mode": parse_mode
                }
                
                response = requests.post(url, files=files, data=data, timeout=30)
                if response.status_code == 200:
                    result = response.json()
                    if result.get('ok'):
                        self.last_notification_time = time.time()
                        logger.info("Telegram photo sent successfully")
                        return True
                    else:
                        logger.error(f"Telegram API error: {result.get('description', 'Unknown error')}")
                        return False
                else:
                    logger.error(f"HTTP error {response.status_code}: {response.text}")
                    return False
        except Exception as e:
            logger.error(f"Error sending Telegram photo: {e}")
            return False
    
    def send_crash_alert(self, frame_id: int, video_source: str = "", 
                        screenshot_path: Optional[str] = None,
                        track_ids: Optional[list] = None,
                        confidence: Optional[float] = None,
                        additional_info: Optional[Dict] = None) -> bool:
        """
        Kirim crash alert ke Telegram dengan format yang informatif
        
        Args:
            frame_id: Frame number dimana crash terdeteksi
            video_source: Source video (YouTube URL atau file path)
            screenshot_path: Path ke screenshot frame crash (optional)
            track_ids: List track IDs yang terlibat (optional)
            confidence: Confidence score (optional)
            additional_info: Dict dengan info tambahan (optional)
        
        Returns:
            True jika berhasil, False jika gagal
        """
        if not self.enabled:
            return False
        
        # Format message
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        message = f"🚨 <b>CRASH DETECTED!</b>\n\n"
        message += f"📹 <b>Video:</b> {video_source}\n"
        message += f"🎬 <b>Frame:</b> {frame_id}\n"
        message += f"🕐 <b>Time:</b> {timestamp}\n"
        
        if track_ids:
            track_str = ", ".join([f"Track {tid}" for tid in track_ids])
            message += f"🚗 <b>Tracks:</b> {track_str}\n"
        
        if confidence is not None:
            message += f"📊 <b>Confidence:</b> {confidence:.2%}\n"
        
        if additional_info:
            message += f"\n<b>Details:</b>\n"
            for key, value in additional_info.items():
                message += f"  • {key}: {value}\n"
        
        # Kirim message
        success = self.send_message(message)
        
        # Kirim screenshot jika ada
        if screenshot_path and Path(screenshot_path).exists():
            caption = f"Crash detected at Frame {frame_id}\n{timestamp}"
            self.send_photo(screenshot_path, caption=caption)
        
        return success
    
    def send_test_message(self) -> bool:
        """Kirim test message untuk verifikasi"""
        message = "🧪 <b>Test Notification</b>\n\n"
        message += "Jika Anda menerima pesan ini, berarti Telegram notification service berfungsi dengan baik! ✅"
        return self.send_message(message)


if __name__ == "__main__":
    # Test notification service
    print("Testing Telegram Notification Service...")
    
    notifier = TelegramNotifier()
    
    # Test 1: Send text message
    print("\n1. Sending test message...")
    success = notifier.send_test_message()
    if success:
        print("   ✓ Test message sent!")
    else:
        print("   ✗ Failed to send test message")
    
    # Test 2: Send crash alert (without screenshot)
    print("\n2. Sending crash alert (text only)...")
    success = notifier.send_crash_alert(
        frame_id=1234,
        video_source="test_video.mp4",
        track_ids=[12, 32],
        confidence=0.95,
        additional_info={"IoU": "0.933", "Energy Loss": "99%"}
    )
    if success:
        print("   ✓ Crash alert sent!")
    else:
        print("   ✗ Failed to send crash alert")

