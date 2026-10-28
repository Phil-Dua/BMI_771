OH DEER - REVISED PROJECT PROPOSAL

CONTENTS
- oh_deer_revised.pdf: two-page CVPR-style camera-ready proposal (1 page of proposal + references on page 2)
- oh_deer_revised.tex: editable LaTeX source
- oh_deer_references.bib: verified reference metadata, with 11 selected sources

BEFORE SUBMITTING
1. Replace [Team Member 1], [Team Member 2], [email1], [email2] in the .tex file with actual author names and addresses; add any other team members.
2. Confirm the Canvas deadline. The displayed 2026 deadline and the pasted 2025 brief conflict.
3. Confirm the dataset includes camera-site IDs, capture timestamps, and class labels needed for the proposed site-held-out split and season ablation. The proposal explicitly treats their availability as conditional.
4. Verify the Kaggle F1 aggregation code before reporting that score. The project uses attribute-wise macro-F1 as an independent diagnostic.
5. Confirm any requirements for the team's final entry and optional extra ecological labels.

LATEX EDITING
- Download the official CVPR author kit: https://github.com/cvpr-org/author-kit/releases
- Place oh_deer_revised.tex and oh_deer_references.bib in the same folder as the official cvpr.sty.
- Compile with pdflatex, BibTeX, pdflatex, pdflatex, in that order.
- Camera-ready is set by \usepackage{cvpr} with no review/page-numbers option.

The supplied PDF was rendered locally with the CVPR 2026 camera-ready typography and layout measurements; the editable source is intended for use within the official CVPR author kit.

HIGHLIGHTS OF REVISION
- Retained original scientific title and all four required proposal sections.
- Correctly describes supplied animal bounding boxes as detector-generated, not human-verified ground truth.
- Prioritizes ResNet-50, DINOv2, and BioCLIP 2, with zero-shot prompts only on suitable CLIP-like vision-language models.
- Compares whole images, detector crops, and padded crops.
- Replaces raw site-ID fusion with seasonal metadata, and checks performance when that metadata is absent.
- Uses burst-aware evaluation and site-held-out evaluation where the metadata supports it.
- Separates official leaderboard F1 from attribute-wise macro-F1 and includes latency.
- Limits additional snow prediction to a stretch goal contingent on extra labels.
- Replaced unsupported source attribution for >100M images with a Wisconsin DNR 2025 source; corrected Newbolt 2019 page range to 527-536; replaced 2024 preprint of Vyskocil/Picek with the 2025 proceedings citation; reduced bibliography from 14 to 11 entries.
