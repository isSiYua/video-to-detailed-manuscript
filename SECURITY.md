# Security

## Do not report secrets in public issues

Never paste API keys, Bilibili Cookie headers, `SESSDATA`, Feishu credentials,
server addresses, SSH keys, generated private notes, or raw task artifacts into
a GitHub issue or pull request.

If a credential was committed or posted publicly, revoke/rotate it first, then
remove it from Git history. Deleting only the latest file is not sufficient.

## Local secret storage

- Store runtime secrets outside the repository in a service-manager secret
  store or a mode-`600` environment file.
- Use a dedicated low-risk Bilibili account.
- Never commit Obsidian Vaults, task registries, exported ZIP files, source
  media, ASR caches, or model weights.

## Vulnerability reports

Until a private security contact is configured for the repository, avoid
publishing exploit details. Open a minimal issue asking the maintainer for a
private contact channel without including sensitive reproduction data.
