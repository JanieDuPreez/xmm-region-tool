# Terminal output safety

Interactive CLI output is a display surface, not an authority surface. Paths, SAS diagnostics and other externally derived strings can contain POSIX control characters, so supported console commands render those characters visibly rather than sending them directly to a terminal emulator.

## Rendering contract

`safe_terminal_text()` preserves ordinary printable text and escapes C0/C1 controls, including:

- ESC as `\x1b`;
- embedded newline as `\n`;
- carriage return as `\r`;
- tab as `\t`;
- other C0/C1 values as visible hexadecimal escapes.

The installed `xmm-region` command wraps delegated progress/stdout/stderr writes at the terminal boundary. Structural line endings emitted by `print()` remain real newlines, while line breaks embedded inside an untrusted path or error string become visible escape text. Batch status lines and checker/validation error lines use the same rendering helper explicitly.

This prevents a filename or external SAS message from injecting ANSI escape sequences, fake status lines, terminal-title commands or carriage-return rewrites into supported interactive output.

## Data remains unchanged

Terminal escaping is deliberately **not** applied to:

- workflow or batch manifests;
- projection/provenance sidecars;
- scientific identity material;
- filesystem paths used for I/O;
- subprocess argument arrays;
- internally generated ESAS command strings.

In particular, copy/paste ESAS command construction continues to use the existing `shlex.join()` semantics. Only the final terminal rendering is escaped if the command contains control characters.

Machine-readable evidence therefore retains its original semantics. Users should still review diagnostic evidence before sharing it publicly; terminal escaping is not redaction or anonymization.

## Relationship to bounded subprocess diagnostics

This policy is separate from `docs/subprocess_diagnostics.md`:

- subprocess diagnostic capture limits **how much** external stdout/stderr can be retained;
- terminal rendering controls **how retained text is displayed** interactively.

A bounded string can still contain ESC or newline characters, and a safely rendered string can still contain sensitive local paths. Both protections are required for their respective purposes.

## Validation

Regression tests include filenames/messages containing ESC, newline, carriage return, tab and terminal-control-like sequences. They require no raw terminal control bytes in captured supported CLI output while preserving ordinary filenames/messages and trusted line layout.

No real-SAS rerun is required because this changes rendering only; SAS invocation, detector geometry, provenance and scientific output are unchanged.
