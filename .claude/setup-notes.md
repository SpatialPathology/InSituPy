# Developer setup notes

## `.log/` symlink or junction

`.log/` is excluded from version control (`.gitignore`) and backed up via a private folder
synced through Nextcloud (or any other private cloud). On a fresh clone, `.log/` will be
missing and must be recreated as a link pointing to that private folder - either a symlink
or a directory junction.

**Option A - Symlink (requires admin privileges or Developer Mode):**

1. Create (or locate) the target folder in your cloud sync directory, e.g.:
   `<Nextcloud-root>\insitupy\.log`
2. Open PowerShell **as Administrator** (or enable Developer Mode to skip elevation) and run:
   ```powershell
   New-Item -ItemType SymbolicLink `
     -Path "<repo-root>\.log" `
     -Target "<cloud-sync-root>\insitupy\.log"
   ```
3. Verify: `Get-Item "<repo-root>\.log" | Select-Object LinkType, Target`

**Option B - Directory junction (no admin rights needed, local paths only):**

If you don't have admin rights and don't want to enable Developer Mode, a junction works
identically for this purpose as long as the cloud sync folder is on a local drive (not a
network/UNC path):

```powershell
New-Item -ItemType Junction `
  -Path "<repo-root>\.log" `
  -Target "<cloud-sync-root>\insitupy\.log"
```

Verify the same way: `Get-Item "<repo-root>\.log" | Select-Object LinkType, Target`.

Replace `<repo-root>` with the path to this repository and `<cloud-sync-root>` with the root
of your personal cloud sync folder.

If the sync root ever changes (e.g. the folder is moved), the link will point at a
now-missing path and must be recreated: delete the stale link with
`(Get-Item -Force "<repo-root>\.log").Delete()` (deletes only the link, not its target
contents), then recreate it with the new target path using either option above.

## `planning/` symlink or junction

`planning/` holds the shared backlog (`planning/backlog.md`) as a visible folder so that
note-taking tools which skip dot-folders (e.g. Obsidian) can index it. Like `.log/`, it is
excluded from version control (`.gitignore`) and backed up via the same private cloud sync
folder. On a fresh clone, `planning/` will be missing and must be recreated as a link
pointing to that private folder, the same way as `.log/` above.

**Option A - Symlink (requires admin privileges or Developer Mode):**

1. Create (or locate) the target folder in your cloud sync directory, e.g.:
   `<Nextcloud-root>\insitupy\planning`
2. Open PowerShell **as Administrator** (or enable Developer Mode to skip elevation) and run:
   ```powershell
   New-Item -ItemType SymbolicLink `
     -Path "<repo-root>\planning" `
     -Target "<cloud-sync-root>\insitupy\planning"
   ```
3. Verify: `Get-Item "<repo-root>\planning" | Select-Object LinkType, Target`

**Option B - Directory junction (no admin rights needed, local paths only):**

If you don't have admin rights and don't want to enable Developer Mode, a junction works
identically for this purpose as long as the cloud sync folder is on a local drive (not a
network/UNC path):

```powershell
New-Item -ItemType Junction `
  -Path "<repo-root>\planning" `
  -Target "<cloud-sync-root>\insitupy\planning"
```

Verify the same way: `Get-Item "<repo-root>\planning" | Select-Object LinkType, Target`.

Replace `<repo-root>` with the path to this repository and `<cloud-sync-root>` with the root
of your personal cloud sync folder.

If the sync root ever changes (e.g. the folder is moved), the link will point at a
now-missing path and must be recreated: delete the stale link with
`(Get-Item -Force "<repo-root>\planning").Delete()` (deletes only the link, not its target
contents), then recreate it with the new target path using either option above.
