"""
alerts.py
Sends an email when a defective item is logged. Runs in a background thread so the
web page never waits for the mail server. Does nothing if SMTP settings are empty.
"""
import smtplib
import threading
from email.message import EmailMessage

from config import SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASS, ALERT_TO


def alerts_enabled() -> bool:
    return bool(SMTP_USER and SMTP_PASS and ALERT_TO)


def _send(subject: str, body: str):
    try:
        msg = EmailMessage()
        msg["Subject"] = subject
        msg["From"] = SMTP_USER
        msg["To"] = ALERT_TO
        msg.set_content(body)
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=15) as server:
            server.starttls()
            server.login(SMTP_USER, SMTP_PASS)
            server.send_message(msg)
        print("Alert email sent to", ALERT_TO)
    except Exception as exc:
        print("Alert email failed:", exc)


def send_defect_alert(record: dict):
    if not alerts_enabled():
        print("Email alerts disabled (SMTP settings empty).")
        return
    body = (
        "A defective product was detected.\n\n"
        f"Time       : {record['inspected_at']}\n"
        f"Image      : {record['image_name']}\n"
        f"Defect type: {record['defect_type']}\n"
        f"Confidence : {record['confidence']}%\n"
        f"Source     : {record['source']}\n"
        f"Operator   : {record['operator']}\n\n"
        f"Reason: {record['reason']}\n"
    )
    threading.Thread(target=_send, args=("[VQC] Defective product detected", body), daemon=True).start()
