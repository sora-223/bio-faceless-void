---
name: bio-faceless-void
description: Train and validate a 2D binary U-Net on single-channel microscopy sections with animal-level splits. Use when a user wants to audit image-mask pairs, segment fluorescent signal, evaluate held-out animals, predict masks on new sections, or quantify a labeled region per animal. Accepts Chinese requests about U-Net训练、切片分割、荧光阳性面积和敲除鼠图像定量.
---

# bio-Faceless Void

Resolve `scripts/unet.py` relative to this SKILL.md. Run it with the user's configured Python interpreter. Resolve manifest and output locations in the user's project. When running outside the skill directory, use absolute paths to the script and manifest. Read [README.md](README.md) for the CSV contract, environment setup and output definitions.

1. Establish which single-channel signal should count as foreground and what tissue area supplies the denominator. Use an agreed ROI at a comparable anatomical level. With VGLUT2 labeling, describe VGLUT2-positive signal. Distinguish this from Draxin localization, axon counts, growth trajectories and a developmental mechanism.
2. Inspect the supplied files. For training, obtain paired human annotations at the original pixel dimensions and a manifest containing animal ID and split. For prediction, obtain compatible trained weights and a two-column manifest. If the user has only unlabeled images and no compatible weights, identify the missing annotation or checkpoint. Use the synthetic demo only when the user wants to try the software. Never invent IDs, labels, or subgroup assignments.
3. Run `audit` before training. Keep every field, crop and serial section from one animal in the same split. Address missing pairs, duplicate pixels and mixed labels. Preserve genuine negative fields. If only one animal is available, describe a technical demonstration and request independent animals for generalization. Check group and batch coverage from the available metadata. Avoid tuning on the test set.
4. Train with the confirmed raw-intensity scale and a fresh output directory. Set tile size to a multiple of eight. Reduce batch size first if GPU memory is insufficient. Only use horizontal flips when anatomical orientation permits. Select the saved checkpoint by validation Dice. Report missing inputs or a failed run accurately.
5. Use `evaluate` with the selected checkpoint on held-out animals with manual masks. Inspect original/overlay pairs including dim structures and negative fields. Report foreground Dice and IoU alongside false positives and false negatives. If predictions miss the structure, revise labels or training settings using development data and preserve a fresh independent test for the revised analysis.
6. Use `predict` for new unlabeled images acquired under comparable conditions. Return the checkpoint path, result tables, representative overlays and any excluded images. State the animal count, ROI denominator and aggregation. Keep segmentation agreement separate from wet-lab conclusions. Without a real labeled test set, report only that the software ran.

For a compact reproducibility check run `python tests/test_workflow.py -v` from the skill directory. Never execute course archives or download study data merely to run the demonstration.

