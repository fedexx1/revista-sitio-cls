"""Paragraph-level TEI parser for the SITIO corpus.

Emits one row per <tei:p> across the 5 issue files, with the entity layer
(persName/placeName refs) and the concept layer (@ana) populated where present.

XInclude resolution + the trivial helpers (NS, get_root, clean_id) are
adapted from revista-sitio-digital/visualizations/tei_parser.py:18-31 to
avoid taking on that project's package-relative imports.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

import lxml.etree as ET
import pandas as pd

# Default to the monorepo layout (this file at 2026/projects/messh/sitio_full/src/).
# A standalone clone sets SITIO_TEI_DIR to point at the revista-sitio-digital TEI dir.
_DEFAULT_TEI_DIR = (
    Path(__file__).resolve().parents[4]
    / "projects" / "revista-sitio-digital" / "TEI"
)
TEI_DIR = Path(os.environ["SITIO_TEI_DIR"]) if os.environ.get("SITIO_TEI_DIR") else _DEFAULT_TEI_DIR
OUT_DIR = Path(__file__).resolve().parents[1] / "data"

ISSUES = ["issue_1", "issue_2", "issue_3", "issue_4-5", "issue_6"]
NS = {"tei": "http://www.tei-c.org/ns/1.0"}
XML_ID = "{http://www.w3.org/XML/1998/namespace}id"


def get_root(path: Path) -> ET._Element:
    """Load a TEI XML file with XInclude resolution (standoff files merged)."""
    tree = ET.parse(str(path))
    tree.xinclude()
    return tree.getroot()


def clean_id(ref: str | None) -> str | None:
    if not ref:
        return None
    return ref.lstrip("#").strip() or None


def _enclosing_typed_div(p_elem: ET._Element) -> ET._Element | None:
    """Innermost ancestor <div> carrying @type."""
    for anc in p_elem.iterancestors(f"{{{NS['tei']}}}div"):
        if anc.get("type"):
            return anc
    return None


def _refs_excluding_notes(p_elem: ET._Element, tag: str) -> list[str]:
    nodes = p_elem.xpath(
        f".//tei:{tag}[@ref][not(ancestor::tei:note)]", namespaces=NS
    )
    out = []
    for n in nodes:
        # TEI @ref is data.pointer+ : a space-separated list of pointers
        # (e.g. ref="#kafka #joyce"). Tokenize, mirroring the @ana handling below.
        for tok in (n.get("ref") or "").split():
            cid = clean_id(tok)
            if cid:
                out.append(cid)
    return out


def _text_identity_div(p_elem: ET._Element) -> ET._Element | None:
    """Nearest ancestor <div> with an xml:id; else the OUTERMOST ancestor div."""
    outer = None
    for anc in p_elem.iterancestors(f"{{{NS['tei']}}}div"):
        if anc.get(XML_ID):
            return anc
        outer = anc
    return outer


def _contributor(p_elem: ET._Element) -> str | None:
    """Byline/signed author of the enclosing text. Search ancestors innermost-out for a
    div carrying a direct byline (or signed/closer-signed); prefer persName@ref inside it,
    else normalized text."""
    for anc in p_elem.iterancestors(f"{{{NS['tei']}}}div"):
        nodes = (anc.xpath("./tei:byline", namespaces=NS)
                 or anc.xpath("./tei:signed | ./tei:closer/tei:signed", namespaces=NS))
        if not nodes:
            continue
        node = nodes[0]
        refs = node.xpath(".//tei:persName/@ref", namespaces=NS)
        if refs:
            cid = clean_id(str(refs[0]).split()[0])
            if cid:
                return cid
        txt = re.sub(r"\s+", " ", "".join(node.itertext())).strip().lower()
        txt = re.sub(r"^por\s+", "", txt)
        return txt or None
    return None


def parse_corpus(tei_dir: Path = TEI_DIR) -> pd.DataFrame:
    if not tei_dir.is_dir():
        raise FileNotFoundError(
            f"TEI source directory not found: {tei_dir}\n"
            f"Set SITIO_TEI_DIR to the revista-sitio-digital/TEI directory."
        )
    rows = []
    for issue in ISSUES:
        root = get_root(tei_dir / f"{issue}.xml")
        for p in root.xpath(".//tei:p", namespaces=NS):
            div = _enclosing_typed_div(p)
            text = re.sub(r"\s+", " ", "".join(p.itertext())).strip()
            ana = p.get("ana") or ""
            ana_tokens = [clean_id(t) for t in ana.split() if clean_id(t)]
            tdiv = _text_identity_div(p)
            text_id = ((tdiv.get(XML_ID) or f"{issue}_pos{int(tdiv.sourceline or 0)}")
                       if tdiv is not None else f"{issue}_orphan")
            rows.append({
                "para_id": p.get(XML_ID),
                "issue": issue,
                "div_id": div.get(XML_ID) if div is not None else None,
                "div_type": div.get("type") if div is not None else None,
                "div_subtype": div.get("subtype") if div is not None else None,
                "text_id": text_id,
                "contributor": _contributor(p),
                "text": text,
                "persons": _refs_excluding_notes(p, "persName"),
                "places": _refs_excluding_notes(p, "placeName"),
                "ana_gold": ana_tokens,
                "has_note": bool(p.xpath(".//tei:note", namespaces=NS)),
            })
    return pd.DataFrame(rows)


def parse_concepts(issue_1_path: Path) -> dict[str, str]:
    root = get_root(issue_1_path)
    out = {}
    for interp in root.xpath(".//tei:interpGrp/tei:interp", namespaces=NS):
        cid = interp.get(XML_ID)
        label = re.sub(r"\s+", " ", "".join(interp.itertext())).strip()
        if cid:
            out[cid] = label
    return out


def parse_persons(list_person_path: Path) -> pd.DataFrame:
    root = get_root(list_person_path)
    rows = []
    for person in root.xpath("//tei:person", namespaces=NS):
        pid = person.get(XML_ID)
        if not pid:
            continue
        # A persName may carry several <forename> elements (given + middle names).
        # Prefer @type='given' if tagged, else keep all forenames in document order.
        given = person.xpath(".//tei:forename[@type='given']/text()", namespaces=NS)
        forename = given or person.xpath(".//tei:forename/text()", namespaces=NS)
        surname = person.xpath(".//tei:surname/text()", namespaces=NS)
        if forename and surname:
            parts = [str(f).strip() for f in forename] + [str(surname[0]).strip()]
            name = " ".join(p for p in parts if p)
        elif surname:
            name = str(surname[0]).strip()
        else:
            full = "".join(person.xpath(".//tei:persName//text()", namespaces=NS))
            name = re.sub(r"\s+", " ", full).strip()
        rows.append({"id": pid, "name": name})
    return pd.DataFrame(rows)


def parse_note_stats(tei_dir: Path = TEI_DIR) -> dict:
    """Corpus-wide <note> structural counts used by the audit's robustness section.

    Computes the figures the robustness argument depends on instead of hardcoding
    them: total notes, the editor commentary layer (resp~="#ED"), how many of those
    sit inside a <p> (the structural contamination path), and total notes inside <p>.
    """
    n_notes = n_ed = n_ed_in_p = n_notes_in_p = 0
    for issue in ISSUES:
        root = get_root(tei_dir / f"{issue}.xml")
        for note in root.xpath(".//tei:note", namespaces=NS):
            n_notes += 1
            in_p = bool(note.xpath("ancestor::tei:p", namespaces=NS))
            if in_p:
                n_notes_in_p += 1
            resp = note.get("resp") or ""
            if "#ED" in resp.split() or resp == "#ED":
                n_ed += 1
                if in_p:
                    n_ed_in_p += 1
    return {
        "n_notes": n_notes,
        "n_ed": n_ed,
        "n_ed_in_p": n_ed_in_p,
        "n_notes_in_p": n_notes_in_p,
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = parse_corpus()
    df.to_parquet(OUT_DIR / "paragraphs.parquet", index=False)
    concepts = parse_concepts(TEI_DIR / "issue_1.xml")
    (OUT_DIR / "concepts.json").write_text(
        json.dumps(concepts, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    persons = parse_persons(TEI_DIR / "listPerson.xml")
    persons.to_csv(OUT_DIR / "persons.csv", index=False, encoding="utf-8")
    note_stats = parse_note_stats()
    (OUT_DIR / "note_stats.json").write_text(
        json.dumps(note_stats, indent=2), encoding="utf-8"
    )

    print(f"paragraphs.parquet: {len(df)} rows")
    print(f"  with ana_gold: {(df['ana_gold'].str.len() > 0).sum()}")
    print(f"  issues with ana tags: "
          f"{sorted(df.loc[df['ana_gold'].str.len() > 0, 'issue'].unique().tolist())}")
    print(f"concepts.json: {len(concepts)} defined concepts")
    print(f"persons.csv: {len(persons)} persons")
    print(f"note_stats.json: {note_stats}")


if __name__ == "__main__":
    main()


# --- Prosopography derivation (SP2). Rule tables copied verbatim from
# revista-sitio-digital/visualizations/config.py (HISTORICAL_PERIODS / COUNTRY_MAPPINGS /
# REGION_MAPPINGS) so sitio_full reproduces the SITIO bucketing without that repo. ---
HISTORICAL_PERIODS = [
    (-800, -500, "Antiquity (Archaic)"), (-500, -300, "Classical Antiquity"),
    (-300, 500, "Hellenistic/Roman"), (500, 1400, "Medieval"), (1400, 1600, "Renaissance"),
    (1600, 1789, "Early Modern"), (1789, 1848, "Revolutionary Era"),
    (1848, 1914, "Long 19th Century"), (1914, 1945, "World Wars Era"), (1945, 2000, "Contemporary"),
]

COUNTRY_MAPPINGS = {
    "argentina": "Argentina", "france": "France", "francia": "France", "germany": "Germany",
    "alemania": "Germany", "prusia": "Germany", "austria": "Austria", "england": "United Kingdom",
    "inglaterra": "United Kingdom", "reino unido": "United Kingdom", "ireland": "Ireland",
    "irlanda": "Ireland", "italy": "Italy", "italia": "Italy", "spain": "Spain", "espania": "Spain",
    "españa": "Spain", "usa": "United States", "estados unidos": "United States", "russia": "Russia",
    "rusia": "Russia", "poland": "Poland", "polonia": "Poland", "greece": "Greece", "grecia": "Greece",
    "switzerland": "Switzerland", "suiza": "Switzerland", "czech": "Czech Republic",
    "praga": "Czech Republic", "cuba": "Cuba", "mexico": "Mexico", "méxico": "Mexico",
    "uruguay": "Uruguay", "chile": "Chile", "brasil": "Brazil", "brazil": "Brazil",
    "venezuela": "Venezuela", "peru": "Peru", "perú": "Peru", "colombia": "Colombia",
    "japan": "Japan", "japón": "Japan", "china": "China", "israel": "Israel",
    "netherlands": "Netherlands", "países bajos": "Netherlands", "belgium": "Belgium",
    "bélgica": "Belgium", "portugal": "Portugal", "hungary": "Hungary", "hungría": "Hungary",
    "romania": "Romania", "rumanía": "Romania", "denmark": "Denmark", "dinamarca": "Denmark",
    "sweden": "Sweden", "suecia": "Sweden", "norway": "Norway", "noruega": "Norway",
}

REGION_MAPPINGS = {
    "Argentina": "Latin America", "Mexico": "Latin America", "Cuba": "Latin America",
    "Uruguay": "Latin America", "Chile": "Latin America", "Brazil": "Latin America",
    "Venezuela": "Latin America", "Peru": "Latin America", "Colombia": "Latin America",
    "France": "Western Europe", "Germany": "Western Europe", "United Kingdom": "Western Europe",
    "Italy": "Western Europe", "Spain": "Western Europe", "Austria": "Western Europe",
    "Switzerland": "Western Europe", "Netherlands": "Western Europe", "Belgium": "Western Europe",
    "Ireland": "Western Europe", "Portugal": "Western Europe", "Greece": "Southern Europe / Classical",
    "United States": "North America", "Russia": "Eastern Europe", "Poland": "Eastern Europe",
    "Czech Republic": "Eastern Europe", "Hungary": "Eastern Europe", "Romania": "Eastern Europe",
    "Israel": "Middle East", "Japan": "Asia", "China": "Asia", "Denmark": "Scandinavia",
    "Sweden": "Scandinavia", "Norway": "Scandinavia",
}


def _first(xs):
    return xs[0] if xs else None


def _year(when):
    """Year int from a TEI @when ('1871-10-30', '1883', '-0075' BCE). None if unparseable."""
    if not when:
        return None
    s = str(when).strip()
    neg = s.startswith("-")
    digits = (s[1:] if neg else s).split("-")[0]
    if not digits.isdigit():
        return None
    return -int(digits) if neg else int(digits)


def _period(year):
    if year is None:
        return "Unknown"
    for lo, hi, name in HISTORICAL_PERIODS:
        if lo <= year < hi:
            return name
    return "Unknown"


def _place_country_map(list_places_path: Path) -> dict:
    """place xml:id -> canonical country (via listPlaces2 location/country, normalized)."""
    root = get_root(list_places_path)
    out = {}
    for pl in root.xpath("//tei:place", namespaces=NS):
        pid = pl.get(XML_ID)
        ctext = pl.xpath("./tei:location/tei:country/text()", namespaces=NS)
        if pid and ctext:
            raw = str(ctext[0]).strip()
            out[pid] = COUNTRY_MAPPINGS.get(raw.lower(), raw)
    return out


def parse_prosopography(list_person_path: Path, list_places_path: Path) -> pd.DataFrame:
    """Per-person demographics: id, name, birth_year, death_year, country, region, period, wikidata."""
    names = parse_persons(list_person_path).set_index("id")["name"].to_dict()
    place_country = _place_country_map(list_places_path)
    root = get_root(list_person_path)
    rows = []
    for person in root.xpath("//tei:person", namespaces=NS):
        pid = person.get(XML_ID)
        if not pid:
            continue
        by = _year(_first(person.xpath(".//tei:birth/tei:date/@when", namespaces=NS)))
        dy = _year(_first(person.xpath(".//tei:death/tei:date/@when", namespaces=NS)))
        ref = _first(person.xpath(".//tei:birth/tei:placeName/@ref", namespaces=NS))
        country = place_country.get(ref.lstrip("#")) if ref else None
        region = REGION_MAPPINGS.get(country, "Other/Unknown") if country else "Other/Unknown"
        wd = _first(person.xpath(".//tei:idno[@subtype='wikidata']/text()", namespaces=NS))
        rows.append({"id": pid, "name": names.get(pid, ""), "birth_year": by, "death_year": dy,
                     "country": country, "region": region, "period": _period(by),
                     "wikidata": str(wd) if wd else None})
    return pd.DataFrame(rows)
