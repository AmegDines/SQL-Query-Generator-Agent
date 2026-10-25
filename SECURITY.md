# Security

## Reporting a vulnerability

Do not report vulnerabilities in public issues. Contact the repository owner privately through GitHub's security reporting feature.

## Before pushing

- Keep `.env` and all real credentials out of Git. Use `.env.example` with blank or clearly fake values for configuration documentation.
- Do not commit database files, Chroma data, checkpoints, generated reports, logs, local credentials, or private keys. The repository ignore rules cover these paths and common extensions.
- Review the complete staged change list and scan it for credentials before every push. Ignore rules do not remove files that were already staged or committed.
- If a credential was committed, revoke or rotate it immediately. Removing it in a later commit does not make the exposed credential safe.

## GitHub repository settings

For repositories on a plan that supports them, enable secret scanning and push protection. Require pull requests and passing checks on the default branch, restrict who can push directly, and require multifactor authentication for collaborators. These settings are configured on GitHub and are not controlled by files in this folder.
