# Rugby fields

A personnal computer vision project with the ultimate goal of counting the number of rugby fields in 
France.

Made to develop my CV skills, and get more confortable with the most used tools in this sector.

---

## Project Scope

Goal
Detect and localize sports fields in satellite imagery as a first step toward rugby field identification.

Output
Bounding boxes around detected fields.

Success Metrics
- mAP@50 ≥ 0.75
- Recall ≥ 0.90
- Precision ≥ 0.85

Constraints
- No annotated dataset available
- Limited compute resources
- Satellite imagery only
- Small-scale manual labeling

MVP (v1)
Detect sports fields regardless of sport type.

Future Versions
- V2: classify field type (rugby, football, etc.)
- V3: end-to-end rugby field detection and classification