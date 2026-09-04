# Security Policy

## Supported Versions

Security updates and critical patches are actively provided for the following versions:

| Version | Supported | Notes |
| :--- | :---: | :--- |
| **1.1.x** | :white_check_mark: | Current stable release |
| **1.0.x** | :white_check_mark: | Critical security and data-loss fixes only |
| **< 1.0.0** | :x: | Unsupported legacy development versions |

---

## Security Architecture & Design Principles

Photo Curator handles user filesystem assets and executes deep learning inference. As a result, the following core security guarantees are architected into every release:

1. **Transactional Two-Phase Commit (`copy_top_images`)**:
   - Files are never directly written or moved into the destination directory. All new images and manifest entries are prepared in a dedicated temporary staging directory (`.curator_stage_*`).
   - If an error, filesystem exception, or I/O failure occurs during copying or manifest writing, changes are rolled back to the prior state and staging directories are completely wiped. *(Note: Asynchronous process termination signals such as SIGINT/SIGKILL bypass Python exception handlers and are not currently intercepted).*

2. **Path Traversal Protection**:
   - Manifest entries undergo strict path-traversal sanitization. Any entry containing relative path components (`..`, `.`), root paths (`/`), or resolving outside the target album directory is immediately rejected with a `RuntimeError`.

3. **Symlink Attack Mitigation**:
   - The internal curator manifest (`.curator_manifest.json`) is validated to ensure it is a regular file. If a symlink is detected for any manifest or tracked file, execution is immediately aborted to prevent arbitrary file clobbering or symlink redirection attacks.

4. **Untracked File Protection**:
   - Photo Curator strictly refuses to overwrite any existing file in the target directory unless that file is explicitly registered in the previous run's `.curator_manifest.json` or the user explicitly specifies the `--overwrite` flag.
   - Atomic exclusive creation (`os.O_CREAT | os.O_EXCL`) is used to prevent race-condition file collisions during commit.

5. **Memory Management & Safe Image Decoding**:
   - Images are loaded within bounded memory constraints to protect against decompression bomb vulnerabilities (`Image.MAX_IMAGE_PIXELS`).
   - Dimension-capped thumbnailing immediately reduces uncompressed raster buffers down to 1024px (~4.72 MB combined across both PIL RGB and OpenCV BGR buffers per image, or ~151 MB for a batch of 32), preventing multi-gigabyte cumulative batch memory accumulation during multi-photo scoring. Note that initial file decompression and EXIF transposition still temporarily require full uncompressed raster memory proportional to raw image dimensions (~569 MiB transient peak RSS for an 8000×6000 JPEG).

---

## Reporting a Vulnerability

We take the security of Photo Curator and its users seriously. If you believe you have found a security vulnerability in this project, please follow these steps:

1. **Do NOT file a public issue.**
2. Report the vulnerability privately via **GitHub Security Advisories**:
   - Navigate to [https://github.com/bhargavkukadiya/photo-curator/security/advisories](https://github.com/bhargavkukadiya/photo-curator/security/advisories)
   - Click **"Report a vulnerability"** to submit your findings confidentially.
   - Alternatively, contact the maintainer directly via email at `bhargavkukadiya@users.noreply.github.com` with the subject `[SECURITY] Photo Curator Vulnerability Report`.
3. Provide detailed steps to reproduce the vulnerability, including:
   - Operating system and Python version
   - A minimal proof of concept (PoC) or script
   - Potential impact and recommended mitigation, if known

### Response Timeline

- **Initial Response:** Within 48 hours acknowledging receipt of your report.
- **Triage & Status Update:** Within 5 business days with an assessment of severity and remediation plan.
- **Disclosure:** We coordinate disclosure with you once a security fix has been released.
