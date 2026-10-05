# Temperature optimum (`opt`) dataset

Statistics of `train.csv`, `val.csv` and `test.csv` in this folder (8,347 / 987 / 1,037 rows; 10,371 in total). Columns: `accession`, `ec`, `organism`, `temperature_optimum` (°C), `sequence`.

## 1. Temperature distribution (°C)
| Split | n | mean | SD | median (IQR) | min–max | < 25 | 25–50 | 50–80 | ≥ 80 |
|---|---|---|---|---|---|---|---|---|---|
| train | 8,347 | 39.14 | 15.89 | 37 (30–45) | 4–120 | 746 | 5,710 | 1,559 | 332 |
| val | 987 | 39.18 | 15.83 | 37 (30–45) | 4–100 | 88 | 675 | 185 | 39 |
| test | 1,037 | 39.15 | 15.76 | 37 (30–45) | 4–100 | 93 | 709 | 194 | 41 |

Temperature distributions of validation and test match train: two-sample KS distance 0.01 (val) and 0.009 (test).

Rows per 10 °C bin:

| bin (°C) | train | val | test |
|---|---|---|---|
| 0–9 | 21 | 2 | 3 |
| 10–19 | 55 | 7 | 7 |
| 20–29 | 1779 | 210 | 221 |
| 30–39 | 3838 | 454 | 476 |
| 40–49 | 763 | 90 | 95 |
| 50–59 | 733 | 87 | 91 |
| 60–69 | 508 | 60 | 63 |
| 70–79 | 318 | 38 | 40 |
| 80–89 | 200 | 24 | 25 |
| 90–99 | 110 | 13 | 14 |
| 100–109 | 19 | 2 | 2 |
| 110–119 | 1 | 0 | 0 |
| 120 | 2 | 0 | 0 |

## 2. EC class (first digit)
| EC class | train | val | test | total |
|---|---|---|---|---|
| 1 | 1927 | 228 | 239 | 2,394 |
| 2 | 2088 | 247 | 260 | 2,595 |
| 3 | 2740 | 324 | 340 | 3,404 |
| 4 | 767 | 91 | 95 | 953 |
| 5 | 459 | 54 | 57 | 570 |
| 6 | 263 | 31 | 33 | 327 |
| 7 | 103 | 12 | 13 | 128 |

All seven EC classes are present in every split. EC-class composition is the same across splits (Cramér's V = 0.001). The dataset has 2,932 distinct EC numbers; 808 of 987 validation rows (81.9 %) and 854 of 1,037 test rows (82.4 %) have an EC number that also occurs in train.

## 3. Organisms
| Domain | train | val | test |
|---|---|---|---|
| Bacteria | 3602 | 453 | 476 |
| Eukaryota | 3698 | 409 | 422 |
| Archaea | 741 | 86 | 98 |
| Viruses | 296 | 37 | 40 |
| Unresolved | 10 | 2 | 1 |

2,563 distinct organism names (train 2,236, val 524, test 545). Most frequent: *Homo sapiens* (700 train rows, 72 val, 85 test), *Arabidopsis thaliana*, *Escherichia coli*, *Mus musculus*, *Saccharomyces cerevisiae*. 798 of 987 validation rows (80.9 %) and 864 of 1,037 test rows (83.3 %) come from an organism that is also in train. Domains were assigned from the UniProt/NCBI taxonomy of the organism name.

## 4. Sequences
| Split | length median (IQR) | length min–max | < 32 aa | > 1,000 aa | archived-sequence rows | secondary-accession rows |
|---|---|---|---|---|---|---|
| train | 410 (306–545) | 5–4,613 | 8 | 306 | 770 | 60 |
| val | 407 (317–560) | 10–7,096 | 2 | 54 | 100 | 9 |
| test | 394 (302–531) | 10–4,017 | 3 | 40 | 102 | 6 |

- The model reads at most 1,000 residues; the 400 sequences longer than 1,000 are truncated. 13 sequences are shorter than 32 residues.
- *Archived-sequence rows* (972, 9.4 %) use the last UniSave sequence of a UniProt entry that has since been deleted (940) or demerged (32). *Secondary-accession rows* (75) have an accession that UniProt now lists as secondary; their sequence is that of the current primary entry. Archived rows are about 9 to 10 % of each split, secondary-accession rows under 1 %. Both groups are listed in [`flag_note.txt`](flag_note.txt).
