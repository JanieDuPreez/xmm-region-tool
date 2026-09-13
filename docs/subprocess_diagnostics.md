# SAS subprocess diagnostic evidence

`xmm-region-tool` executes SAS tasks as argument arrays and treats their stdout/stderr as **execution diagnostics**, not as portable scientific provenance.

## Capture policy

Package-owned SAS subprocess calls use one bounded capture policy for:

- `esky2det -v` producer discovery;
- `sasversion` producer discovery;
- science-coordinate projection through `esky2det`;
- real-SAS REGION validation through `evselect`.

Each stdout and stderr stream is directed to a temporary file while the child process runs. Python therefore does not accumulate an unbounded pipe buffer in memory. After the process exits, at most 64 KiB per stream is retained as diagnostic evidence. If a stream exceeds that ceiling, the retained value contains its head and tail separated by an explicit truncation marker, and evidence records include:

- total bytes emitted;
- bytes retained from the original stream;
- whether truncation occurred.

Short diagnostics below the ceiling are preserved unchanged apart from normal text newline/UTF-8 handling.

The human-facing exception message uses a smaller excerpt of the already bounded diagnostic text. This prevents a large retained evidence window from being duplicated into error-message state or batch manifests.

## Producer/version probes

Producer identity must not be inferred from incomplete version output. If either output stream from `esky2det -v` or `sasversion` exceeds the diagnostic ceiling, producer discovery fails explicitly even if a recognizable version string appears in the retained head or tail.

This is intentionally stricter than ordinary failed-task diagnostics: a truncated failed `esky2det`/`evselect` log can still be useful execution evidence, while a truncated version probe is not accepted as complete producer-identification authority.

## Privacy and portability

SAS diagnostics can contain absolute local paths, SAS environment details, filenames, or other machine-specific runtime information. The bounded capture policy limits retention; it does **not** redact or anonymize the retained text.

Accordingly:

- stdout/stderr are execution evidence and may be unsuitable for public sharing without user review;
- they are not inputs to scientific projection identity;
- they are not claimed to be path-independent or portable provenance;
- batch `failure_evidence` may contain the bounded retained diagnostic excerpts and their truncation metadata.

Interactive control-character escaping/sanitization is a separate terminal-rendering concern tracked independently from this retention policy. Bounding a stream and making it safe to render are different operations.

## Runtime policy

This hardening deliberately adds **no arbitrary SAS execution timeout**. Scientific SAS runtime can depend strongly on event size, calibration state and task workload. A future watchdog policy would require its own evidence-based contract rather than using a security/resource issue to invent a scientific runtime ceiling.

The output-size bound changes only diagnostic retention. Successful projection geometry, SAS arguments, calibration identity and producer identity semantics are unchanged.
