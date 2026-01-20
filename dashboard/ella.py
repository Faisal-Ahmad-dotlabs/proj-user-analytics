from concurrent.futures import ThreadPoolExecutor, as_completed
import functools
import logging
from typing import Any

import requests
import pandas as pd

from . import config

logger = logging.getLogger(__name__)
settings = config.get_app_settings()


@functools.lru_cache
def _prepare_headers(env: str) -> dict[str, str]:
    if env == 'dev':
        token = settings.ELLA_KEY_TEST.get_secret_value()
    else:
        token = settings.ELLA_KEY_PROD.get_secret_value()

    headers = {"Authorization": f"Bearer {token}"}

    return headers


@functools.lru_cache
def fetch_content_details(env: str, content_id: str) -> dict[str, Any]:
    if pd.isna(content_id) or content_id == "":
        return {
            "title": "Unknown",
            "word_count": 0,
            "type": "",
            "upload_date":
            "",
        }

    if env == 'dev':
        base_url = settings.ELLA_URL_TEST
    else:
        base_url = settings.ELLA_URL_PROD

    headers = _prepare_headers(env)

    try:
        url = f"{base_url}/{content_id}"
        r = requests.get(url, headers=headers, timeout=10)

        if r.status_code == 200:
            data = r.json()
            result = {
                "title": data.get("content_title", f"Content {str(content_id)[:8]}"),
                "word_count": data.get("content_words", 0),
                "type": data.get("content_type", ""),
                "upload_date": data.get("created_at", ""),
            }
            return result

        result = {
            "title": f"Content {str(content_id)[:8]}...",
            "word_count": 0,
            "type": f"API {r.status_code}",
            "upload_date": "",
        }
        return result
    except Exception:
        logger.exception("fetch content: failed")
        result = {
            "title": f"Content {str(content_id)[:8]}...",
            "word_count": 0,
            "type": "-",
            "upload_date": "",
        }
        return result


def fetch_multiple_contents(env: str, content_ids: list[str]) -> dict[str, str]:
    results: dict[str, Any] = {}

    valid = [cid for cid in content_ids if pd.notna(cid) and cid != ""]
    if not valid:
        return results

    with ThreadPoolExecutor(max_workers=10) as ex:
        futs = {ex.submit(fetch_content_details, env, cid): cid for cid in valid}
        for fut in as_completed(futs):
            cid = futs[fut]
            try:
                results[cid] = fut.result()
            except Exception:
                logger.exception("fetch content (multiple): failed")
                results[cid] = {
                    "title": f"Content {str(cid)[:8]}...",
                    "word_count": 0,
                    "type": "-",
                    "upload_date": "",
                }

    return results
