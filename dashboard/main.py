import logging

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from . import dasher, mixpanel


logging.basicConfig(
    format="%(asctime)s:%(levelname)s:%(name)s:%(message)s",
    level=logging.DEBUG,
)

logger = logging.getLogger(__name__)


def main():
    logger.info('User Analytics Dashboard')
    dasher.app.run(host="0.0.0.0", port=8000, debug=True)


if __name__ == '__main__':
    main()
