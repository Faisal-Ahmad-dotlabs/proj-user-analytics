import base64
import json
import logging

import pandas as pd
import requests

from . import config

logger = logging.getLogger(__name__)


class NoResultError(RuntimeError):
    """Expected a result from an external service, but got none."""
    pass


def _prepare_headers(env: str) -> dict[str, str]:
    settings = config.get_app_settings()

    if env == "prod":
        api_key = settings.MIXPANEL_KEY_PROD.get_secret_value()
    else:
        api_key = settings.MIXPANEL_KEY_TEST.get_secret_value()
    
    auth = base64.b64encode(f"{api_key}".encode()).decode()

    headers = {
        "Authorization": f"Basic {auth}",
        "Accept": "text/plain",
        "User-Agent": "ella-dashboard/1.0",
    }

    return headers


def fetch_mixpanel_data(from_date: str, to_date: str, env: str) -> pd.DataFrame:
    """Fetch data from Mixpanel."""

    hosts = [
        "https://data.mixpanel.com/api/2.0/export",
        "https://data-eu.mixpanel.com/api/2.0/export",
    ]

    headers = _prepare_headers(env)
    # Inside mixpanel.py -> fetch_mixpanel_data
    params = {
        "from_date": from_date, 
        "to_date": to_date,
        "project_id": "3381875" # Use your project ID here
    }

    last_err = None

    for url in hosts:
        try:
            r = requests.get(url, headers=headers, params=params, timeout=300)
            if r.status_code != 200:
                last_err = f"{url} -> {r.status_code}: {r.text[:300]}"
                continue

            text = (r.text or "").strip()
            if not text:
                return pd.DataFrame()

            lines = [ln for ln in text.split('\n') if ln.strip()]
            events = [json.loads(line) for line in lines]
            df_local = pd.json_normalize(events)
            return df_local

        except Exception as e:
            logger.exception(f"mixpanel: {url=}")
            raise

    last_err = last_err or "unknown error"
    raise NoResultError(f"mixpanel: {last_err}")
