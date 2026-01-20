import functools

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    MIXPANEL_KEY_TEST: SecretStr
    MIXPANEL_KEY_PROD: SecretStr

    ELLA_KEY_TEST: SecretStr
    ELLA_KEY_PROD: SecretStr

    ELLA_URL_TEST: str
    ELLA_URL_PROD: str

    PROXY_PATH: str = ''

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="",
    )


@functools.lru_cache
def get_app_settings() -> Settings:
    return Settings()
