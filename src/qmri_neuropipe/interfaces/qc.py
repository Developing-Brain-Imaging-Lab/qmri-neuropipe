
from pathlib import Path
import json, csv
from datetime import datetime

def write_sidecar(target: Path, meta: dict):
    sc = target.with_suffix(target.suffix + ".json")
    sc.parent.mkdir(parents=True, exist_ok=True)
    sc.write_text(json.dumps(meta, indent=2), encoding="utf-8")

class RunLedger:
    def __init__(self, base: Path):
        self.base = base; self.base.mkdir(parents=True, exist_ok=True)
        self.path = self.base / "ledger.csv"
        if not self.path.exists():
            self.path.write_text("started_at,finished_at,dataset,sub,ses,pipeline,status,error\n", encoding="utf-8")
        self.started_at = None
    def start_run(self): self.started_at = datetime.utcnow().isoformat() + "Z"
    def mark_subject(self, sub, ses, status, error=""):
        self.path.write_text(self.path.read_text() + f"{self.started_at},,{''},{sub},{ses or ''},dmri-dti,{status},{error}\n", encoding="utf-8")
    def finish_run(self):
        lines = self.path.read_text().splitlines()
        if len(lines)<=1: return
        hdr, *rows = lines
        now = datetime.utcnow().isoformat() + "Z"
        new = [hdr]
        for r in rows:
            parts = r.split(",")
            if parts[1]=="":
                parts[1]=now
                new.append(",".join(parts))
            else:
                new.append(r)
        self.path.write_text("\n".join(new)+"\n", encoding="utf-8")
