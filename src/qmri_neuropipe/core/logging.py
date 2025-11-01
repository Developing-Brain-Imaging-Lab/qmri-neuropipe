
import json, logging, sys
from datetime import datetime
from pathlib import Path

log = logging.getLogger("qmri-neuropipe")

class JsonFormatter(logging.Formatter):
    def format(self, record):
        payload = {
            "ts": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "name": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload)

def init_logging(output_dir: Path, verbose: bool = False):
    log.setLevel(logging.DEBUG if verbose else logging.INFO)
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(JsonFormatter())
    log.handlers.clear()
    log.addHandler(h)
    (output_dir / "derivatives" / "qmri-neuropipe" / "logs").mkdir(parents=True, exist_ok=True)
