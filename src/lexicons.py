"""Hand-curated lexicons used by audit.py.

Two lists:
- BLANCHOT_SEEDS: Blanchotian conceptual vocabulary. Whole-word match,
  case-insensitive, no stemming. Derived from sitio_publication/results/
  blanchot_analysis/blanchot_vocabulary_analysis.json (top concepts found
  in Issue 1) plus the canonical "afuera / dehors" terms.

- BLANCHOT_CIRCLE: the seven authors Blanchot wrote critical essays on,
  as listPerson xml:ids. Per the accepted MeSSH abstract.
  IDs verified against persons.csv at audit run time; missing ids fail loud.
"""

BLANCHOT_SEEDS = [
    "escritura", "escribir", "escrito", "escrita",
    "ausencia", "ausente", "afuera", "dehors",
    "errancia", "errar",
    "muerte", "olvido", "silencio", "soledad", "imposible",
]

BLANCHOT_CIRCLE = [
    "kafka",
    "james_joyce",
    "samuel_beckett",
    "louis_ferdinand_celine",
    "william_faulkner",
    "borges",
    "witold_gombrowicz",
]

BLANCHOT_ID = "maurice_blanchot"

# Whole-word, case-insensitive surname match on paragraph text. Used to build the
# stratified-test exclusion set symmetrically with group A (also text-derived), so
# that footnote mentions of Blanchot are excluded too — not just persName tags
# outside notes.
BLANCHOT_NAME = "blanchot"
