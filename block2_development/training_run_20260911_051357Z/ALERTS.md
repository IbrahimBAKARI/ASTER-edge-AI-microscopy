
## WARNING — 07:34:16Z

1 x40 field(s) unreadable, excluded from the stress test

```
train/slide_20260908_162808_658.jpg: broken data stream when reading image file
```

## WARNING — 10:51:15Z

['abn_promy_frac'] cannot support a proportion claim (TPR - FPR < 0.05). Any rule resting on them returns indeterminate by construction - report it, do not retune.

## FAILED — 12:34:30Z

uncaught exception: AttributeError: 'SessionResult' object has no attribute 'n_classified'

```
Traceback (most recent call last):
  File "/usr/local/lib/python3.13/dist-packages/IPython/core/interactiveshell.py", line 3553, in run_code
    exec(code_obj, self.user_global_ns, self.user_ns)
    ~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/tmp/ipykernel_2925/1822017992.py", line 10, in <cell line: 0>
    "flags": "|".join(result.flags), "n_classified": result.n_classified,
                                                     ^^^^^^^^^^^^^^^^^^^
AttributeError: 'SessionResult' object has no attribute 'n_classified'
```

## FAILED — 12:35:19Z

uncaught exception: AttributeError: 'SessionResult' object has no attribute 'n_classified'

```
Traceback (most recent call last):
  File "/usr/local/lib/python3.13/dist-packages/IPython/core/interactiveshell.py", line 3553, in run_code
    exec(code_obj, self.user_global_ns, self.user_ns)
    ~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/tmp/ipykernel_2925/1822017992.py", line 10, in <cell line: 0>
    "flags": "|".join(result.flags), "n_classified": result.n_classified,
                                                     ^^^^^^^^^^^^^^^^^^^
AttributeError: 'SessionResult' object has no attribute 'n_classified'
```

## FAILED — 15:19:02Z

uncaught exception: AssertionError: frozen arm differs from the primary run on 9 patients - stop

```
Traceback (most recent call last):
  File "/usr/local/lib/python3.13/dist-packages/IPython/core/interactiveshell.py", line 3553, in run_code
    exec(code_obj, self.user_global_ns, self.user_ns)
    ~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/tmp/ipykernel_2925/2009999316.py", line 47, in <cell line: 0>
    assert not mismatch, f"frozen arm differs from the primary run on {len(mismatch)} patients - stop"
           ^^^^^^^^^^^^
AssertionError: frozen arm differs from the primary run on 9 patients - stop
```

## FAILED — 15:19:04Z

uncaught exception: AssertionError: frozen arm differs from the primary run on 9 patients - stop

```
Traceback (most recent call last):
  File "/usr/local/lib/python3.13/dist-packages/IPython/core/interactiveshell.py", line 3553, in run_code
    exec(code_obj, self.user_global_ns, self.user_ns)
    ~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/tmp/ipykernel_2925/2009999316.py", line 47, in <cell line: 0>
    assert not mismatch, f"frozen arm differs from the primary run on {len(mismatch)} patients - stop"
           ^^^^^^^^^^^^
AssertionError: frozen arm differs from the primary run on 9 patients - stop
```

## FAILED — 15:19:05Z

uncaught exception: AssertionError: frozen arm differs from the primary run on 9 patients - stop

```
Traceback (most recent call last):
  File "/usr/local/lib/python3.13/dist-packages/IPython/core/interactiveshell.py", line 3553, in run_code
    exec(code_obj, self.user_global_ns, self.user_ns)
    ~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/tmp/ipykernel_2925/2009999316.py", line 47, in <cell line: 0>
    assert not mismatch, f"frozen arm differs from the primary run on {len(mismatch)} patients - stop"
           ^^^^^^^^^^^^
AssertionError: frozen arm differs from the primary run on 9 patients - stop
```

## RESOLUTIONS — reviewed 2026-09-11

- 07:34 x40 unreadable field: expected; excluded and named; stress test ran on the 614 crops of the remaining fields.
- 10:51 abn_promy_frac unquantifiable: declared limitation; R1a returns indeterminate by construction; reported, not retuned.
- 12:34, 12:35 AttributeError n_classified (section 9): notebook bug (SessionResult names it number_of_classified_leukocytes); fixed, section 9 re-run, 409/409 patients scored after the fix.
- 15:19 (logged x3) frozen arm differs on 9 patients (section 11): an interrupted legacy section-11 cell had left gamma at 0.80. The assertion stopped the cell before any result was written; gamma restored to 0.90, replay 409/409, section 11 re-run. No reported result was computed with the drifted value.

## FAILED — 16:30:13Z

uncaught exception: TypeError: unsupported format string passed to NoneType.__format__

```
Traceback (most recent call last):
  File "/usr/local/lib/python3.13/dist-packages/IPython/core/interactiveshell.py", line 3553, in run_code
    exec(code_obj, self.user_global_ns, self.user_ns)
    ~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/tmp/ipykernel_2925/11667312.py", line 15, in <cell line: 0>
    print(f"epoch {r.get('epoch'):>3}  train {r.get('loss'):.4f}  val AUROC {r.get('mil_val_auroc'):.4f}  val loss {r.get('mil_val_loss'):.4f}")
                                                                                                                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^
TypeError: unsupported format string passed to NoneType.__format__
```

## RESOLUTIONS — reviewed 2026-09-11

- 07:34 x40 unreadable field: expected; excluded and named; stress test ran on the 614 crops of the remaining fields.
- 10:51 abn_promy_frac unquantifiable: declared limitation; R1a returns indeterminate by construction; reported, not retuned.
- 12:34, 12:35 AttributeError n_classified (section 9): notebook bug (SessionResult names it number_of_classified_leukocytes); fixed, section 9 re-run, 409/409 patients scored after the fix.
- 15:19 (logged x3) frozen arm differs on 9 patients (section 11): an interrupted legacy section-11 cell had left gamma at 0.80. The assertion stopped the cell before any result was written; gamma restored to 0.90, replay 409/409, section 11 re-run. No reported result was computed with the drifted value.

## FAILED — 16:30:45Z

uncaught exception: TypeError: unsupported format string passed to NoneType.__format__

```
Traceback (most recent call last):
  File "/usr/local/lib/python3.13/dist-packages/IPython/core/interactiveshell.py", line 3553, in run_code
    exec(code_obj, self.user_global_ns, self.user_ns)
    ~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/tmp/ipykernel_2925/11667312.py", line 15, in <cell line: 0>
    print(f"epoch {r.get('epoch'):>3}  train {r.get('loss'):.4f}  val AUROC {r.get('mil_val_auroc'):.4f}  val loss {r.get('mil_val_loss'):.4f}")
                                                                                                                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^
TypeError: unsupported format string passed to NoneType.__format__
```
