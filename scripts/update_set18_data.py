"""Build a validated TFT Set 18 snapshot from CommunityDragon data.

The PBE aggregate can temporarily contain incomplete champion lists.  For that
reason, champions are discovered from the localized client manifest and their
costs/stats are read from the corresponding character records.
"""

from __future__ import annotations

import argparse
import json
import tempfile
import urllib.request
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


COMMUNITY_DRAGON = "https://raw.communitydragon.org"
SET_NUMBER = 18
EXPECTED_UNIT_COUNTS = {18: 65}


def download_json(url: str, timeout: int = 60) -> Any:
    """Download and decode a JSON resource."""
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "tftgenetics-data-updater/1.0"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def load_json(path: Path) -> Any:
    """Load JSON from a local file."""
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def find_character_record(bin_data: dict[str, Any], character_id: str) -> dict[str, Any]:
    """Find the playable character record in a CommunityDragon bin export."""
    records = [
        value
        for value in bin_data.values()
        if isinstance(value, dict)
        and value.get("__type") == "TFTCharacterRecord"
        and value.get("mCharacterName") == character_id
    ]
    if len(records) != 1:
        raise ValueError(
            f"Expected one TFTCharacterRecord for {character_id}, found {len(records)}"
        )
    return records[0]


def scalar(record: dict[str, Any], key: str) -> float | None:
    """Read a baseValue field from a character record."""
    value = record.get(key)
    if isinstance(value, dict):
        value = value.get("baseValue")
    return value if isinstance(value, (int, float)) else None


def discover_set_units(
    manifest: list[dict[str, Any]], set_number: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return playable units and alternate Lux forms for a set."""
    marker = f"tft_set{set_number}.png"
    candidates = [
        entry
        for entry in manifest
        if marker in entry.get("character_record", {}).get("squareIconPath", "").lower()
    ]
    alternate_forms = [
        entry
        for entry in candidates
        if entry.get("character_record", {}).get("display_name", "").startswith("Lux (")
    ]
    playable = [entry for entry in candidates if entry not in alternate_forms]
    return playable, alternate_forms


def build_trait_choices(alternate_forms: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert Lux's alternate client records into one-of trait choices."""
    choices: dict[str, dict[str, Any]] = {}
    for form in alternate_forms:
        for trait in form["character_record"]["traits"]:
            if trait["name"] == "Avatar":
                continue
            choices[trait["id"]] = {
                "traitApiName": trait["id"],
                "traitName": trait["name"],
                "count": 2,
            }
    return sorted(choices.values(), key=lambda choice: choice["traitName"])


def build_units(
    manifest: list[dict[str, Any]],
    alternate_forms: list[dict[str, Any]],
    channel: str,
    set_number: int,
    character_cache: Path,
) -> list[dict[str, Any]]:
    """Build normalized units, enriching the manifest with character records."""
    units = []
    lux_choices = build_trait_choices(alternate_forms)
    character_cache.mkdir(parents=True, exist_ok=True)

    for entry in manifest:
        source = entry["character_record"]
        character_id = source["character_id"]
        cache_file = character_cache / f"{character_id.lower()}.json"
        url = (
            f"{COMMUNITY_DRAGON}/{channel}/game/characters/"
            f"{character_id.lower()}.cdtb.bin.json"
        )
        if cache_file.exists():
            bin_data = load_json(cache_file)
        else:
            bin_data = download_json(url)
            with cache_file.open("w", encoding="utf-8") as file:
                json.dump(bin_data, file, ensure_ascii=False)

        record = find_character_record(bin_data, character_id)
        traits = [
            {
                "apiName": trait["id"],
                "name": trait["name"],
                "count": 2
                if character_id == "DA_18_ElderDragon" and trait["name"] == "Riftbeast"
                else 1,
            }
            for trait in source["traits"]
        ]
        unit = {
            "apiName": character_id,
            "name": source["display_name"],
            "cost": record["tier"],
            "teamSlots": 2 if character_id == "DA_18_ElderDragon" else 1,
            "traits": traits,
            "stats": {
                "health": scalar(record, "baseHPModifiable"),
                "attackDamage": scalar(record, "baseDamageModifiable"),
                "armor": scalar(record, "baseArmorModifiable"),
                "magicResist": scalar(record, "baseMR"),
                "attackSpeed": scalar(record, "attackSpeedModifiable"),
                "attackRange": scalar(record, "attackRangeModifiable"),
            },
            "icon": source["squareIconPath"],
        }
        if character_id == "DA_Lux18_Base":
            unit["traitChoice"] = {
                "type": "chooseOne",
                "options": lux_choices,
            }
        units.append(unit)

    return sorted(units, key=lambda unit: (unit["cost"], unit["name"]))


def build_traits(
    raw_traits: list[dict[str, Any]], units: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Build traits and reverse references to their native units."""
    contributions: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for unit in units:
        for trait in unit["traits"]:
            contributions[trait["apiName"]].append(
                {"unit": unit["apiName"], "count": trait["count"]}
            )

    traits = []
    for raw_trait in raw_traits:
        thresholds = sorted(
            {
                effect["minUnits"]
                for effect in raw_trait.get("effects", [])
                if isinstance(effect.get("minUnits"), int) and effect["minUnits"] > 0
            }
        )
        trait = {
            "apiName": raw_trait["apiName"],
            "name": raw_trait["name"],
            "thresholds": thresholds,
            "effects": raw_trait.get("effects", []),
            "description": raw_trait.get("desc", ""),
            "units": sorted(
                contributions.get(raw_trait["apiName"], []),
                key=lambda contribution: contribution["unit"],
            ),
            "icon": raw_trait.get("icon"),
        }
        if raw_trait["apiName"] == "DA_18_Eclipse":
            trait["activation"] = {
                "type": "allTraitsActive",
                "traits": ["DA_18_Solar", "DA_18_Lunar"],
            }
        traits.append(trait)
    return sorted(traits, key=lambda trait: trait["name"])


def build_emblems(
    items: list[dict[str, Any]], traits: list[dict[str, Any]], set_number: int
) -> list[dict[str, Any]]:
    """Normalize and deduplicate emblem items by granted trait."""
    prefix = f"DA_{set_number}_Emblem"
    raw_emblems = [item for item in items if item.get("apiName", "").startswith(prefix)]
    traits_by_name = {trait["name"]: trait for trait in traits}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in raw_emblems:
        trait_name = item["name"].removesuffix(" Emblem")
        if trait_name not in traits_by_name:
            raise ValueError(f"Cannot map emblem {item['apiName']} to trait {trait_name}")
        grouped[trait_name].append(item)

    emblems = []
    for trait_name, sources in grouped.items():
        recipes = sorted(
            {
                tuple(source.get("composition", []))
                for source in sources
                if source.get("composition")
            }
        )
        emblems.append(
            {
                "name": f"{trait_name} Emblem",
                "traitApiName": traits_by_name[trait_name]["apiName"],
                "traitName": trait_name,
                "craftable": bool(recipes),
                "recipes": [list(recipe) for recipe in recipes],
                "sourceApiNames": sorted(source["apiName"] for source in sources),
            }
        )
    return sorted(emblems, key=lambda emblem: emblem["traitName"])


def validate_snapshot(snapshot: dict[str, Any]) -> None:
    """Reject incomplete or internally inconsistent snapshots."""
    set_number = snapshot["set"]["number"]
    units = snapshot["units"]
    traits = snapshot["traits"]
    emblems = snapshot["emblems"]
    expected = EXPECTED_UNIT_COUNTS.get(set_number)
    if expected is not None and len(units) != expected:
        raise ValueError(f"Set {set_number}: expected {expected} units, found {len(units)}")

    unit_ids = [unit["apiName"] for unit in units]
    trait_ids = [trait["apiName"] for trait in traits]
    if len(unit_ids) != len(set(unit_ids)):
        raise ValueError("Duplicate unit apiName")
    if len(trait_ids) != len(set(trait_ids)):
        raise ValueError("Duplicate trait apiName")
    trait_id_set = set(trait_ids)
    unit_id_set = set(unit_ids)

    for unit in units:
        if unit["cost"] not in range(1, 6):
            raise ValueError(f"Invalid cost for {unit['apiName']}: {unit['cost']}")
        for trait in unit["traits"]:
            if trait["apiName"] not in trait_id_set:
                raise ValueError(
                    f"Unknown trait {trait['apiName']} on unit {unit['apiName']}"
                )
    for trait in traits:
        if trait["thresholds"] != sorted(set(trait["thresholds"])):
            raise ValueError(f"Invalid thresholds for {trait['apiName']}")
        for contribution in trait["units"]:
            if contribution["unit"] not in unit_id_set:
                raise ValueError(
                    f"Unknown unit {contribution['unit']} on trait {trait['apiName']}"
                )
    for emblem in emblems:
        if emblem["traitApiName"] not in trait_id_set:
            raise ValueError(f"Unknown emblem trait {emblem['traitApiName']}")

    counts = Counter(unit["cost"] for unit in units)
    if set_number == 18 and counts != Counter({1: 14, 2: 13, 3: 14, 4: 14, 5: 10}):
        raise ValueError(f"Unexpected Set 18 cost distribution: {dict(counts)}")


def create_snapshot(
    set_number: int,
    channel: str,
    raw_tft: dict[str, Any],
    champion_manifest: list[dict[str, Any]],
    character_cache: Path,
) -> dict[str, Any]:
    """Create a complete normalized set snapshot."""
    set_data = raw_tft["sets"][str(set_number)]
    playable, alternate_forms = discover_set_units(champion_manifest, set_number)
    units = build_units(
        playable,
        alternate_forms,
        channel,
        set_number,
        character_cache,
    )
    traits = build_traits(set_data["traits"], units)
    emblems = build_emblems(raw_tft["items"], traits, set_number)
    snapshot = {
        "schemaVersion": 2,
        "set": {
            "number": set_number,
            "name": f"TFT Set {set_number}",
            "channel": channel,
        },
        "source": {
            "provider": "CommunityDragon",
            "generatedAt": datetime.now(timezone.utc).isoformat(),
            "tftData": f"{COMMUNITY_DRAGON}/{channel}/cdragon/tft/en_us.json",
            "championManifest": (
                f"{COMMUNITY_DRAGON}/{channel}/plugins/rcp-be-lol-game-data/"
                "global/default/v1/tftchampions.json"
            ),
        },
        "traits": traits,
        "units": units,
        "emblems": emblems,
    }
    validate_snapshot(snapshot)
    return snapshot


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--channel", choices=("pbe", "latest"), default="pbe")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/TFTSet18_full_lookup.json"),
    )
    parser.add_argument("--tft-data", type=Path)
    parser.add_argument("--champion-manifest", type=Path)
    parser.add_argument("--character-cache", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tft_url = f"{COMMUNITY_DRAGON}/{args.channel}/cdragon/tft/en_us.json"
    champion_url = (
        f"{COMMUNITY_DRAGON}/{args.channel}/plugins/rcp-be-lol-game-data/"
        "global/default/v1/tftchampions.json"
    )
    raw_tft = load_json(args.tft_data) if args.tft_data else download_json(tft_url)
    champion_manifest = (
        load_json(args.champion_manifest)
        if args.champion_manifest
        else download_json(champion_url)
    )

    if args.character_cache:
        snapshot = create_snapshot(
            SET_NUMBER,
            args.channel,
            raw_tft,
            champion_manifest,
            args.character_cache,
        )
    else:
        with tempfile.TemporaryDirectory(prefix="tftgenetics-") as directory:
            snapshot = create_snapshot(
                SET_NUMBER,
                args.channel,
                raw_tft,
                champion_manifest,
                Path(directory),
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as file:
        json.dump(snapshot, file, ensure_ascii=False, indent=2)
        file.write("\n")
    print(
        f"Wrote {args.output}: {len(snapshot['units'])} units, "
        f"{len(snapshot['traits'])} traits, {len(snapshot['emblems'])} emblems"
    )


if __name__ == "__main__":
    main()
