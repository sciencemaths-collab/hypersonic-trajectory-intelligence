# Security Policy

## Scope

This repository is public research software. Do not use the public repository to store credentials, private telemetry, controlled technical data, proprietary datasets, or sensitive operational information.

## Reporting a vulnerability

Use GitHub's private vulnerability reporting / security-advisory workflow for this repository when available. Do not open a public issue containing secrets, exploit details that expose users, or private data.

## Current security controls

- NumPy trace loading uses `allow_pickle=False`.
- Model-state guidance uses `torch.load(..., weights_only=True)` where supported.
- The Streamlit console launches the local engine through an argument-list subprocess rather than shell-string execution.
- CI performs dependency auditing against `requirements.txt`.
- Versioned benchmark evidence is checked for accidental absolute workstation paths.
- Generated model/output artifacts are ignored by default.

## Deployment cautions

The Streamlit application is intended for local research use. Do not expose it directly to the public internet for sensitive workloads without authentication, authorization, TLS termination, request limits, logging controls, and a deployment-specific security review.

Do not treat model outputs as authorization for a physical or safety-critical action.

## Dependency and artifact hygiene

- Prefer pinned/locked environments for release evidence.
- Review dependency-audit failures before release.
- Do not deserialize untrusted Python pickle objects.
- Verify model and evidence checksums when artifacts are transferred between systems.
- Keep external datasets outside the repository unless their license and sensitivity permit publication.

## Commit identity privacy

Git commit metadata is public in a public repository. Contributors who do not want a personal email exposed should configure a GitHub-provided `noreply` commit email before making commits. Removing an email already present in Git history generally requires a deliberate history rewrite and should not be done casually on a shared repository.
