"""
alerts.py - Real-Time Alert Management & Threat Logging Module for IBVAP-Lite
Tracks watchlist matches, deduplicates within cooldown windows, formats alerts,
and generates structured logs and analytics for the security dashboard.
Includes instant Telegram Bot push alerts with forensic evidence photos.
"""

import os
import time
import uuid
import threading
import urllib.request
import urllib.parse
import json
import pandas as pd
from datetime import datetime


def _load_telegram_config():
    """Loads Telegram config from telegram_config.json, .env, or os.environ."""
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.environ.get("TELEGRAM_CHAT_ID", "").strip()

    # Try local telegram_config.json
    config_file = os.path.join(os.path.dirname(__file__), "telegram_config.json")
    if os.path.exists(config_file):
        try:
            with open(config_file, "r", encoding="utf-8") as f:
                cfg = json.load(f)
                if not token and cfg.get("bot_token"):
                    token = str(cfg.get("bot_token")).strip()
                if not chat_id and cfg.get("chat_id"):
                    chat_id = str(cfg.get("chat_id")).strip()
        except Exception:
            pass

    # Try .env file if available
    env_file = os.path.join(os.path.dirname(__file__), ".env")
    if os.path.exists(env_file):
        try:
            with open(env_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("TELEGRAM_BOT_TOKEN=") and not token:
                        token = line.split("=", 1)[1].strip().strip('"').strip("'")
                    elif line.startswith("TELEGRAM_CHAT_ID=") and not chat_id:
                        chat_id = line.split("=", 1)[1].strip().strip('"').strip("'")
        except Exception:
            pass

    return token, chat_id


def save_telegram_config(bot_token, chat_id):
    """Saves Telegram credentials to telegram_config.json."""
    config_file = os.path.join(os.path.dirname(__file__), "telegram_config.json")
    data = {
        "bot_token": str(bot_token).strip(),
        "chat_id": str(chat_id).strip()
    }
    with open(config_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    os.environ["TELEGRAM_BOT_TOKEN"] = data["bot_token"]
    os.environ["TELEGRAM_CHAT_ID"] = data["chat_id"]
    return True


def test_telegram_connection(bot_token=None, chat_id=None):
    """
    Sends a test ping to Telegram to verify credentials and fetches bot/chat metadata.
    Returns: (success: bool, message: str, details: dict)
    """
    cfg_token, cfg_chat = _load_telegram_config()
    token = str(bot_token or cfg_token or "").strip()
    target_chat = str(chat_id or cfg_chat or "").strip()

    if not token:
        return False, "Telegram Bot Token is missing.", {}
    if not target_chat:
        return False, "Telegram Chat ID is missing.", {}

    bot_info = {
        "bot_username": "",
        "bot_name": "",
        "chat_name": target_chat,
        "chat_id": target_chat
    }

    # Fetch bot metadata
    try:
        req_me = urllib.request.Request(f"https://api.telegram.org/bot{token}/getMe")
        with urllib.request.urlopen(req_me, timeout=5.0) as resp:
            data_me = json.loads(resp.read().decode("utf-8"))
            if data_me.get("ok"):
                bot_info["bot_username"] = data_me["result"].get("username", "")
                bot_info["bot_name"] = data_me["result"].get("first_name", "")
    except Exception:
        pass

    # Fetch chat metadata
    try:
        req_chat = urllib.request.Request(f"https://api.telegram.org/bot{token}/getChat?chat_id={target_chat}")
        with urllib.request.urlopen(req_chat, timeout=5.0) as resp:
            data_chat = json.loads(resp.read().decode("utf-8"))
            if data_chat.get("ok"):
                c = data_chat["result"]
                full_name = f"{c.get('first_name', '')} {c.get('last_name', '')}".strip()
                bot_info["chat_name"] = full_name or c.get("title", "") or target_chat
    except Exception:
        pass

    # Send verified test message using HTML
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = json.dumps({
        "chat_id": target_chat,
        "text": (
            "🟢 <b>IBVAP-Lite Border Surveillance</b>\n"
            "━━━━━━━━━━━━━━━━━━\n"
            "🛡️ <b>Telegram Dispatch Connected Successfully!</b>\n"
            "Target face recognition alerts with forensic evidence photos are armed and ready."
        ),
        "parse_mode": "HTML"
    }).encode("utf-8")

    try:
        req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=5.0) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            if data.get("ok"):
                return True, "Test message delivered successfully to Telegram!", bot_info
            return False, f"Telegram API error: {data.get('description', 'Unknown error')}", bot_info
    except urllib.error.HTTPError as he:
        try:
            err_body = he.read().decode("utf-8")
            err_json = json.loads(err_body)
            desc = err_json.get("description", str(he))
        except Exception:
            desc = str(he)
        return False, f"HTTP Error {he.code}: {desc}", bot_info
    except Exception as e:
        return False, f"Connection error: {str(e)}", bot_info


def _dispatch_telegram_alert(alert_record, bot_token=None, chat_id=None):
    """
    Sends an instant push alert with forensic details and cropped face photo to Telegram.
    Runs asynchronously in a background thread to prevent any inference lag.
    """
    import html
    cfg_token, cfg_chat = _load_telegram_config()
    token = str(bot_token or cfg_token or "").strip()
    target_chat = str(chat_id or cfg_chat or "").strip()

    if not token or not target_chat:
        print("[TELEGRAM] Alert skipped: BOT_TOKEN or CHAT_ID not configured.")
        return

    person = alert_record.get("person", "Unknown")
    confidence = alert_record.get("confidence_pct", "0%")
    threat = alert_record.get("threat_level", "HIGH")
    timestamp = alert_record.get("timestamp", "")
    source = alert_record.get("source", "Camera")
    snap_path = alert_record.get("snapshot_path", "")

    # HTML formatted caption — safe against special character parsing crashes
    person_h = html.escape(str(person))
    conf_h = html.escape(str(confidence))
    threat_h = html.escape(str(threat))
    source_h = html.escape(str(source))
    ts_h = html.escape(str(timestamp))

    caption = (
        f"🚨 <b>BORDER ALERT: TARGET IDENTIFIED</b>\n"
        f"━━━━━━━━━━━━━━━━━━\n"
        f"👤 <b>Person:</b> <code>{person_h}</code>\n"
        f"🎯 <b>Confidence:</b> <code>{conf_h}</code>\n"
        f"⚠️ <b>Classification:</b> {threat_h}\n"
        f"📍 <b>Source:</b> <code>{source_h}</code>\n"
        f"⏱️ <b>Timestamp:</b> <code>{ts_h}</code>\n"
        f"━━━━━━━━━━━━━━━━━━\n"
        f"🛡️ <b>IBVAP-Lite Autonomous Command</b>"
    )

    try:
        if snap_path and os.path.exists(snap_path):
            # Send photo with caption using multipart form-data
            url = f"https://api.telegram.org/bot{token}/sendPhoto"
            boundary = f"----WebKitFormBoundary{uuid.uuid4().hex}"
            body = bytearray()

            def add_field(name, value):
                body.extend(f"--{boundary}\r\n".encode())
                body.extend(f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode())
                body.extend(f"{value}\r\n".encode())

            add_field("chat_id", target_chat)
            add_field("caption", caption)
            add_field("parse_mode", "HTML")

            with open(snap_path, "rb") as f:
                photo_bytes = f.read()

            body.extend(f"--{boundary}\r\n".encode())
            body.extend(f'Content-Disposition: form-data; name="photo"; filename="evidence.jpg"\r\n'.encode())
            body.extend(b"Content-Type: image/jpeg\r\n\r\n")
            body.extend(photo_bytes)
            body.extend(b"\r\n")
            body.extend(f"--{boundary}--\r\n".encode())

            req = urllib.request.Request(
                url,
                data=bytes(body),
                headers={"Content-Type": f"multipart/form-data; boundary={boundary}"}
            )
            with urllib.request.urlopen(req, timeout=5.0) as resp:
                print(f"[TELEGRAM] Alert photo dispatched successfully for '{person}'.")
        else:
            # Fallback text message
            url = f"https://api.telegram.org/bot{token}/sendMessage"
            payload = json.dumps({
                "chat_id": target_chat,
                "text": caption,
                "parse_mode": "HTML"
            }).encode("utf-8")
            req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=4.0) as resp:
                print(f"[TELEGRAM] Alert message dispatched successfully for '{person}'.")
    except Exception as e:
        print(f"[WARN] Telegram push failed: {e}")


class AlertManager:
    """
    Manages security threat alerts with cooldown deduplication, analytics,
    and instant remote Telegram notifications.
    """

    def __init__(self, cooldown_seconds=4.0, max_history=250):
        self.cooldown_seconds = cooldown_seconds
        self.max_history = max_history
        self.alerts = []  # List of alert dicts (newest first)
        self.last_alert_times = {}  # {person_name: timestamp}
        self.latest_banner = None
        self.latest_banner_time = 0.0
        self.telegram_token, self.telegram_chat_id = _load_telegram_config()

    def trigger_alert(self, person_name, confidence, source="Live Feed", bbox=None, match_tier="HIGH", snapshot_path=None):
        """
        Records a watchlist match. Applies cooldown deduplication per person per tier.
        - HIGH tier: 5s cooldown (confirmed match)
        - POSSIBLE tier: 12s cooldown (soft warning, logged separately)
        Returns:
            (is_new_alert: bool, alert_data: dict)
        """
        current_time = time.time()

        # Tier-specific cooldown
        cooldown = self.cooldown_seconds if match_tier == "HIGH" else self.cooldown_seconds * 3

        # Separate cooldown key per person + tier so HIGH and POSSIBLE don't block each other
        cooldown_key = f"{person_name}::{match_tier}"
        last_time = self.last_alert_times.get(cooldown_key, 0.0)
        is_cooldown = (current_time - last_time) < cooldown
        self.last_alert_times[cooldown_key] = current_time

        # Update visual banner: only HIGH tier gets the big red banner
        if match_tier == "HIGH":
            self.latest_banner = {
                "person": person_name,
                "confidence": confidence,
                "tier": match_tier,
                "timestamp": datetime.now().strftime("%H:%M:%S")
            }
            self.latest_banner_time = current_time

        if is_cooldown:
            return False, None

        # Threat level label
        if match_tier == "HIGH":
            threat_level = "HIGH — CONFIRMED MATCH"
        elif match_tier == "POSSIBLE":
            threat_level = "MEDIUM — POSSIBLE MATCH (Verify)"
        else:
            threat_level = "LOW — UNCERTAIN"

        # Create new unique alert record
        alert_record = {
            "id": str(uuid.uuid4())[:8],
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "date": datetime.now().strftime("%Y-%m-%d"),
            "person": person_name,
            "confidence_pct": f"{confidence * 100:.1f}%",
            "confidence_raw": round(confidence, 4),
            "threat_level": threat_level,
            "match_tier": match_tier,
            "source": source,
            "snapshot_path": snapshot_path or ""
        }

        # Prepend to keep newest on top
        self.alerts.insert(0, alert_record)
        if len(self.alerts) > self.max_history:
            self.alerts.pop()

        # Dispatch Telegram push alert asynchronously in background thread for confirmed matches
        if match_tier == "HIGH" and (self.telegram_token and self.telegram_chat_id):
            threading.Thread(
                target=_dispatch_telegram_alert,
                args=(alert_record, self.telegram_token, self.telegram_chat_id),
                daemon=True
            ).start()

        return True, alert_record

    def get_active_banner(self, banner_duration=2.5):
        """Returns active banner dict if recent threat occurred within duration."""
        if self.latest_banner and (time.time() - self.latest_banner_time) < banner_duration:
            return self.latest_banner
        return None

    def get_recent_alerts(self, limit=50):
        """Returns the most recent alerts (newest first)."""
        return self.alerts[:limit]

    def get_dataframe(self):
        """Converts recent alerts to pandas DataFrame for Streamlit display."""
        if not self.alerts:
            return pd.DataFrame(columns=["Timestamp", "Person", "Confidence", "Threat Level", "Source"])
        
        df = pd.DataFrame(self.alerts)
        return df[["timestamp", "person", "confidence_pct", "threat_level", "source"]].rename(columns={
            "timestamp": "Timestamp",
            "person": "Matched Person",
            "confidence_pct": "Confidence",
            "threat_level": "Threat Level",
            "source": "Video Source"
        })

    def get_stats(self):
        """Returns summary analytics on threats."""
        total_alerts = len(self.alerts)
        unique_targets = len(set(a["person"] for a in self.alerts))
        highest_conf = max([a["confidence_raw"] for a in self.alerts], default=0.0)
        return {
            "total_alerts": total_alerts,
            "unique_targets": unique_targets,
            "highest_confidence": f"{highest_conf * 100:.1f}%" if highest_conf > 0 else "N/A"
        }

    def clear(self):
        """Clears all alerts."""
        self.alerts.clear()
        self.last_alert_times.clear()
        self.latest_banner = None
