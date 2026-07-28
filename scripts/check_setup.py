"""Validate the initial project setup."""

from quant_equity import __version__
from quant_equity.config import (
    PROJECT_ROOT,
    ensure_project_directories,
    get_random_seed,
    load_config,
)
from quant_equity.logging_config import configure_logging


def main() -> None:
    """Run a basic project setup validation."""
    config = load_config()

    ensure_project_directories()

    logger = configure_logging(
        level=config["runtime"]["log_level"],
        log_file=PROJECT_ROOT / "logs" / "setup.log",
    )

    logger.info("Project configuration loaded successfully.")
    logger.info("Project root: %s", PROJECT_ROOT)
    logger.info("Project version: %s", __version__)
    logger.info("Random seed: %s", get_random_seed(config))
    logger.info(
        "Preferred tabular format: %s",
        config["storage"]["tabular_format"],
    )

    print()
    print("Institutional Quant Equity Research Platform")
    print("------------------------------------------------")
    print(f"Version: {__version__}")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Random seed: {get_random_seed(config)}")
    print("Initial setup: OK")


if __name__ == "__main__":
    main()
