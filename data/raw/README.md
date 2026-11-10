# Raw data

Downloaded files. **Never edited by code** — every transformation writes to
`data/processed/`.

| File | Source | Produced by |
|---|---|---|
| `combined_pubchem_data.csv` | PubChem BioAssay, Gene ID 2064 (HER2/ERBB2) | supplied with course materials |
| `combined_pubchem_data_assay_catalogue.csv` | assay triage for the above | supplied with course materials |
| `egfr_assay_catalogue.csv` | PubChem BioAssay, Gene ID 1956 (EGFR/ERBB1) | `src/02_fetch_egfr.py` |
| `egfr_pubchem_data.csv` | data tables for the kept EGFR assays | `src/02_fetch_egfr.py --download-only` |

The two EGFR files are not committed; regenerate them with:

```bash
python src/02_fetch_egfr.py --catalogue-only   # then review the catalogue by hand
python src/02_fetch_egfr.py --download-only
```
