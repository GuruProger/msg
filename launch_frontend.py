from pathlib import Path
import sys
deps=Path(__file__).resolve().parent/'.ui-deps'
if deps.is_dir():sys.path.insert(0,str(deps))
from dome_web.dashboard_backend import main
if __name__=='__main__':main()
