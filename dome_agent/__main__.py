"""Allow ``python -m dome_agent`` to start the shift agent."""

from .agent import main

if __name__ == "__main__":
    raise SystemExit(main())
