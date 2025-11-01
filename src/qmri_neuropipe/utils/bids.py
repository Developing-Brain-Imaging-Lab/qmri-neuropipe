
from pathlib import Path
from typing import List, Tuple

def select_participants_sessions(bids_dir: Path, participants: List[str]|None, sessions: List[str]|None, skip_validation: bool=False) -> List[Tuple[str, str|None]]:
    out=[]
    for p in sorted(Path(bids_dir).glob("sub-*")):
        if not p.is_dir(): continue
        sub=p.name.split("-",1)[1]
        if participants and sub not in participants: continue
        ses_dirs=sorted(p.glob("ses-*"))
        if ses_dirs:
            for sd in ses_dirs:
                ses=sd.name.split("-",1)[1]
                if sessions and ses not in sessions: continue
                out.append((sub,ses))
        else:
            out.append((sub,None))
    return out
