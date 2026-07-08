# Attribution — legacy confessions/catechisms (original Tier C build)

**Status: these 9 files are NOT used in the redistributable dataset.** The
publishable Tier C is the AI-translated public-domain rebuild in
`data/sources/confessions_pd/` (see `configs/pd_sources.json` and
`docs/PD_RETRANSLATION_SPEC.md`). This document exists only to record the
**probable origin** of each legacy file, per project decision: attribute
rather than leave anonymous, but do not redistribute or use them as
translation input (their PT license status is unconfirmed/likely restricted —
see the copyright audit in `docs/PD_RETRANSLATION_SPEC.md`).

Investigated 2026-07-08. No file carries embedded author/license metadata;
attribution below is **inferred** by matching title/wording against public
web sources, not confirmed directly with any rights holder.

| File | Probable origin | Notes |
|---|---|---|
| `A_Confissao_de_Fe_de_Westminster.pdf` | **Igreja Presbiteriana do Brasil (IPB)** — identical title found at `executivaipb.com.br/arquivos/confissao_de_westminster.pdf` (IPB's official executive site). Same wording also republished uncredited on 10+ Brazilian Presbyterian/Reformed sites (IPB congregations, seminaries — IPFB, FATERGE, IBRVN, Bereianos, IBEL, PIPG). | No embedded metadata (author/date/producer only shows Acrobat Distiller, 2009). IPB is the confessional body that formally adopted the Westminster Standards (Constitution, 1950) — plausible institutional source, but translator/date not confirmed. **Caution:** a separate, clearly commercial/copyrighted PT edition of the same document exists (Alderi Souza de Matos, "A Confissão de Fé de Westminster", Editora Fiel/Antioquia, ISBN 8576222434, sold on Amazon) — unclear if related to this text. |
| `wcf_1647.txt` | Same as above (identical opening header/title) — appears to be a plain-text extraction of the same PDF, not an independent source. | Redundant copy, not a distinct source. |
| `Breve_Catecismo_de_Westminster.pdf` | Unconfirmed. No embedded author. PDF dated 2008 (PDFCreator/Ghostscript). | Same generic "Breve Catecismo de Westminster" title circulates on many Brazilian Reformed sites (monergismo.com and others) without consistent attribution. |
| `westminster_shorter_catechism.txt` | Same as above — plain-text extraction of the same PDF. | Redundant copy. |
| `Catecismo_Maior_de_Westminster.pdf` | Author field explicitly names **Daniel Alves Antunes** (embedded PDF metadata, Word 2007, dated 2009). | The only file with a clear named individual — treat as that person's copyrighted work unless/until they grant a license. |
| `westminster_larger_catechism.txt` | Same as above — plain-text extraction. | Redundant copy. |
| `heidelberg_catechism.txt` | **Monergismo.com** and/or **CPRC** (Covenant Protestant Reformed Church, `cprc.co.uk/languages/heidelberg_portugal`) — matching wording found on both during the copyright audit. | Modern PT translation, no named translator found. |
| `canons_of_dort.txt` | **Monergismo.com** (a related "Breve Esboço dos Cânones de Dort" summary on Monergismo names translator **Felipe Sabino de Araújo Neto** — confirms Monergismo hosts named-translator work in this space, though not necessarily this exact full-canons text) and/or Brazilian denominational sites (IPB Vitória, Igreja Reformada, CPRC). | Multiple near-identical PT versions circulate (IPB congregations, Voltemos ao Evangelho, Justificação pela Fé) — a shared translation tradition, not independently verified as free-licensed. |
| `lcf_1689.txt` | Multiple Brazilian Reformed/Baptist sites carry near-identical wording: **Ligonier Brasil** (`pt.ligonier.org`), Igreja Reformada, Firme Fundamento, PIBBRJ, Seminário Batista Confessional do Brasil. | No single named translator identified; likely a shared/reprinted translation tradition among Brazilian Reformed Baptists. |

## Why this doesn't change the publication decision

Wide, uncredited recirculation across many church/ministry websites is **not**
evidence of public-domain or open-license status — it is equally consistent
with informal, unlicensed copying (common practice for confessional texts in
Brazilian evangelical circles, tolerated but not necessarily authorized). Two
concrete findings argue for continued caution rather than relaxation:

1. **A named, commercially-published PT edition of the WCF exists** (Alderi
   Souza de Matos, sold via Amazon/Editora Fiel) — proof that at least one
   real, identifiable, presumably rights-holding author operates in this
   exact space.
2. **The IPB Constitution (which formally carries this translation) dates to
   1950** — only ~75 years old. Even if this specific file is IPB's own
   institutional translation, Brazilian copyright's life+70-years rule means
   it is very unlikely to be public domain yet (the translator would need to
   have died before ~1955).

No confirmation was sought directly from IPB or any other rights holder (per
project decision, 2026-07-08) — this document records probable origin for
attribution/transparency only. The Tier C corpus that ships is the
AI-translated public-domain rebuild (`data/sources/confessions_pd/`), which
does not depend on resolving this question.
