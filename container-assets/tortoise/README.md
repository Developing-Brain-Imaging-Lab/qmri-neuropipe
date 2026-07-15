# Optional TORTOISE container payload

Place prebuilt Linux TORTOISE binaries in this directory before building the
Docker or Apptainer image.

Expected layout:

```text
container-assets/
  tortoise/
    bin/
      CreateGradientNonlinearityBMatrix
      DIFFPREP
      DRBUDDI
      ...
    lib/
      ...
```

During image build, the contents of `container-assets/tortoise` are copied to
`/opt/tortoise`, and `/opt/tortoise/bin` is added to `PATH`.

The GNL pipeline currently requires at least:

```text
container-assets/tortoise/bin/CreateGradientNonlinearityBMatrix
```

Keep large local binary payloads out of git unless their license permits
redistribution. This README exists so the optional container asset path is
available without committing TORTOISE itself.
