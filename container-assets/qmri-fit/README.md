# Optional qmri-fit container payload

Place the static Linux `qmri-fit` binary from the `qmri_nextgen` project in this
directory before building the Docker or Apptainer image.

Preferred layout:

```text
container-assets/
  qmri-fit/
    bin/
      qmri-fit
```

The build also accepts either of these layouts:

```text
container-assets/qmri-fit/qmri-fit
container-assets/qmri-fit/bin/qmri_fit
container-assets/qmri-fit/qmri_fit
```

During image build, the contents are copied to `/opt/qmri-fit`, and
`/opt/qmri-fit/bin` is added to `PATH`. The build creates both executable names
when possible:

```text
qmri-fit
qmri_fit
```

This lets the DESPOT/mcDESPOT relaxometry pipeline find the binary regardless of
whether callers use the hyphenated `qmri-fit` name or the older `qmri_fit`
underscore name.

Keep large local binary payloads out of git unless their license permits
redistribution. This README exists so the optional container asset path is
available without committing the binary itself.
