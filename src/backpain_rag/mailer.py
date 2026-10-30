"""Email through Amazon SES over SMTP. Standard library only.

boto3 would ship the service catalogue for every AWS API to make one SendEmail
call; smtplib is about forty lines and adds nothing to the bundle. Sending
happens inside the request: a serverless instance can be frozen the moment it
responds, and "send it after we reply" loses the email an account depends on.
"""

from __future__ import annotations

import smtplib
import ssl
from email.message import EmailMessage
from email.utils import formataddr

from .config import get_settings


class MailError(RuntimeError):
    pass


def send(to: str, subject: str, text: str) -> None:
    s = get_settings()
    if not s.mail_enabled:
        raise MailError("email is not configured")
    msg = EmailMessage()
    msg["From"] = formataddr((s.mail_from_name, s.mail_from))
    msg["To"] = to
    msg["Subject"] = subject
    msg.set_content(text)
    ctx = ssl.create_default_context()
    try:
        if s.ses_smtp_port == 465:
            with smtplib.SMTP_SSL(s.ses_smtp_host, 465, context=ctx,
                                  timeout=s.mail_timeout_s) as smtp:
                smtp.login(s.ses_smtp_user, s.ses_smtp_password)
                smtp.send_message(msg)
        else:
            with smtplib.SMTP(s.ses_smtp_host, s.ses_smtp_port, timeout=s.mail_timeout_s) as smtp:
                smtp.starttls(context=ctx)
                smtp.login(s.ses_smtp_user, s.ses_smtp_password)
                smtp.send_message(msg)
    except (smtplib.SMTPException, OSError) as e:
        raise MailError(f"could not send email: {e}") from e


def send_verification(to: str, link: str) -> None:
    send(to, "Confirm your Back Pain Help account",
         f"Confirm your email address to start asking questions:\n\n{link}\n\n"
         "The link lasts three days. If you did not sign up, ignore this email.")


def send_reset(to: str, link: str) -> None:
    send(to, "Reset your Back Pain Help password",
         f"Use this link to choose a new password:\n\n{link}\n\n"
         "It works once and expires in an hour. Resetting signs you out everywhere. "
         "If you did not ask for this, ignore this email.")
