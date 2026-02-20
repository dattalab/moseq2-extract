# Flip Classifier / Crop Size Validation

## Problem

When the extraction `crop_size` doesn't match the flip classifier's expected input size, `get_flips()` in `extract/proc.py` silently catches the `ValueError` from scikit-learn and continues extraction without any flip correction. The user sees a printed warning but extraction completes with subtly wrong output — the mouse orientation is never corrected, which degrades all downstream analysis (PCA, AR-HMM syllable discovery).

The mismatch typically happens because K2 (Kinect v2) flip classifiers expect `(80, 80)` crops while the Azure classifier expects `(120, 120)` crops, and these are configured independently.

## Solution

### 1. Early validation in `extract_wrapper()` (`helpers/wrappers.py`)

Add `validate_flip_classifier(config_data)` that:
- Returns immediately if `flip_classifier` is falsy
- Loads the classifier with `joblib.load()`
- Checks `clf.n_features_` against `crop_size[0] * crop_size[1]`
- Raises `ValueError` with a detailed message on mismatch, explaining:
  - The actual vs expected crop sizes
  - K2 classifiers expect (80, 80), Azure expects (120, 120)
  - How the mismatch likely happened (wrong classifier for camera type)
  - Both fixes: change `crop_size` in config, or re-download the correct classifier

Called at the top of `extract_wrapper()`, after config cleaning (~line 356), before any expensive work.

### 2. Harden `get_flips()` (`extract/proc.py`)

Replace the silent `except ValueError` catch (lines 48-55) with a hard `raise` as a safety net for direct callers.

### 3. Test script

Small script run in `moseq2_app` conda env to verify `joblib.load()` and `n_features_` access works with the actual installed sklearn/joblib versions, and that the validation function correctly passes/raises.
