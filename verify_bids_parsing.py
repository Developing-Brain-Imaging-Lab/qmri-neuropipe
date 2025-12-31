import re
import sys

BIDS_ENTITY_REGEX = re.compile(r'([a-zA-Z0-9]+)-([^\_]+)')

def test(name):
    print(f"Testing: {name}")
    matches = BIDS_ENTITY_REGEX.findall(name)
    print(f"Matches: {matches}")
    entities = {}
    for key, value in matches:
        entities[key] = value
    print(f"Entities: {entities}")

test("sub-01a_ses-01_dwi.nii.gz")
test("sub-Abc01_dwi.nii.gz")
test("sub-01_dwi.nii.gz")
