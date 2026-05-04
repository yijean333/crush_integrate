"""
Telegram Bot Configuration
==========================

Configuration untuk Telegram Bot API integration
JANGAN COMMIT FILE INI KE GIT! (sudah di .gitignore)

Setup:
1. Dapatkan Bot Token dari @BotFather di Telegram
2. Dapatkan Chat ID dari @userinfobot atau getUpdates API
3. Isi credentials di bawah ini
"""

# Telegram Bot Token (dari @BotFather)
TELEGRAM_BOT_TOKEN = "8474386818:AAGyes6LZbsu9RInA_iCtN6s63Thie9en3w"

# Telegram Chat ID (untuk kirim notifikasi)
# Dapatkan dari @userinfobot atau https://api.telegram.org/bot<TOKEN>/getUpdates
TELEGRAM_CHAT_ID = "1423682433"  # Chat ID: Angie (@unclegie)

# Bot Username (optional, untuk info saja)
TELEGRAM_BOT_USERNAME = "@thu_crash_detection_info_bot"

# Notification Settings
NOTIFICATION_ENABLED = True  # Set False untuk disable notifikasi
RATE_LIMIT_SECONDS = 1  # Minimal interval antar notifikasi (detik) - Reduced for collision alerts



