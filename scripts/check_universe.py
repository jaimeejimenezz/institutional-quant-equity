"""Load and summarize the configured equity universe."""

from quant_equity.config import PROJECT_ROOT, load_config
from quant_equity.data import load_universe
from quant_equity.logging_config import configure_logging


def main() -> None:
    """Validate the universe and print a summary."""
    config = load_config()

    logger = configure_logging(
        level=config["runtime"]["log_level"],
        log_file=PROJECT_ROOT / "logs" / "universe.log",
    )

    universe_version = config["universe"]["version"]
    universe = load_universe(universe_version)

    sector_counts = (
        universe.groupby("sector", as_index=False)
        .size()
        .sort_values(
            ["size", "sector"],
            ascending=[False, True],
        )
    )

    logger.info(
        "Universe %s loaded successfully.",
        universe_version,
    )
    logger.info(
        "Companies: %s",
        len(universe),
    )
    logger.info(
        "Unique sectors: %s",
        universe["sector"].nunique(),
    )
    logger.info(
        "Active companies: %s",
        int(universe["is_active"].sum()),
    )

    print()
    print("Institutional Quant Equity Research Platform")
    print("Equity universe validation")
    print("------------------------------------------------")
    print(f"Version: {universe_version}")
    print(f"Companies: {len(universe)}")
    print(f"Sectors: {universe['sector'].nunique()}")
    print(f"Active companies: {int(universe['is_active'].sum())}")
    print()
    print("Companies by sector")
    print("------------------------------------------------")
    print(sector_counts.to_string(index=False))
    print()
    print("Universe validation: OK")


if __name__ == "__main__":
    main()
