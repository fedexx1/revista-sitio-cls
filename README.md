# A Self-Theorizing Corpus: Computational Methods and Their Limits in *Revista SITIO* (1981–1987)

Code and derived data for the MeSSH 2026 paper of the same title.

*Revista SITIO* (Buenos Aires, 1981–1987) is a journal of literary criticism whose central
filiation — to Maurice Blanchot — runs almost entirely without citation. This repository contains
the computational apparatus behind the paper: it reads the journal's 3,476 prose paragraphs through
**three representations** and studies where they agree and diverge.

| | Representation | What it is |
|---|---|---|
| **E** | Distributional embedding | each paragraph as a vector (Qwen3-Embedding-8B; bge-m3 as a replication) |
| **C** | Concept layer | concept-words a language model extracts per paragraph |
| **H** | Human apparatus | the TEI edition's cited persons and interpretive concept-tags |

### The three results

1. **The Blanchot filiation, quantified (§3).** Paragraphs carrying Blanchot's vocabulary sit closer
   in embedding space to the authors his criticism read than to the rest of the corpus
   (Cliff's δ = **+0.427**), robust to controls and recovered, prior-free, in the journal's own citations.
2. **A prior-free cartography (§4).** A concept×citation map — built with no embedding model — recovers
   the geography of the journal's thought and its movement across the 1981–1987 transition.
3. **A documented reception event (§5) and its limit (§6).** A core editor's translation of Blanchot
   anchors a dossier whose title rewrites *Le livre à venir*; and a current language model, given the
   editors' own rubric, reproduces their interpretive tags only by over-naming concepts
   (recall **0.873**, precision **0.242**).

## Layout

```
src/        24 analysis scripts (the E/C/H pipeline; published unmodified)
data/       frozen derived inputs — embedding vectors (.npy, Git LFS), concept layer,
            lemmas, person tables, curated lexicons   (NO raw prose)
results/    computed tables and figures behind every claim in the paper
```

## Data

This repository publishes derived artifacts only — embeddings, the concept layer, lemmas, citation
and concept tables, and figures. It does **not** include the journal's prose. The full text is
published separately as a TEI digital edition:
[hdlabconicet/revista-sitio-digital](https://github.com/hdlabconicet/revista-sitio-digital).

## Citation

```bibtex
@inproceedings{cortes2026sitio,
  author    = {Cort\'es, Federico Gabriel},
  title     = {A Self-Theorizing Corpus: Computational Methods and Their Limits
               in {Revista SITIO} (1981--1987)},
  booktitle = {Proceedings of MeSSH 2026},
  year      = {2026}
}
```


